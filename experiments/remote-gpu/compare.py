"""Stock models vs every tuned version, same audio, one table.

    python experiments/remote-gpu/compare.py [--set tts] [--only NAME ...] [--limit N] [--paired A B]
                                             [--features] [--by-source] [--show N] [--fresh]

Candidates: 'stock-whisper' (small.en), 'stock-parakeet' (the app's default engine, fp32),
'stock-parakeet-int8', 'stock-hotwords' (the app's sherpa-onnx engine with radiology hotwords),
plus each folder in models/ that holds a model/ directory; a folder with parakeet.json is a
Parakeet version (read from data/models/fine_tuned/ once moved into the app) and also gets a
'<version>+hw' row (the same model decoded with hotwords), any other folder a Whisper one.
Raw text per clip comes from decode_store (the Kaggle run files its decodes there), so only
what is missing is decoded here; --fresh decodes everything again. Nothing here touches src/
or the app settings.
Scores the model's raw text (no correction stage) so the model alone is judged, plus the text
after the app's correction stage ("+fix"), which is what the user sees.
'unseen' = clips in a voice the model was never trained on: the tts set's held-out voice
(every 3rd clip, starting at the 2nd) or any clip that is not synthetic. Only the tts set
prints it.
"""
import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import NamedTuple

import soundfile as sf

import decode_store
from config import MODELS
from scripts.eval.corpus import load_set
from scripts.eval.metrics import meaning_slips, term_error_rate, word_error_rate
from src.dictation.asr.engines.parakeet_engine import ParakeetEngine, available_models, model_dir_for


def is_unseen(i, clip):
    return not clip.synthetic or i % 3 == 1


def whisper(**kw):
    from src.dictation.asr.engines.faster_whisper_engine import FasterWhisperEngine
    return FasterWhisperEngine(**kw)


def parakeet(path=None, quantization=None):
    """The app's own ParakeetEngine (fp32 unless told); a tuned folder is loaded into it, so decoding is the app's code path."""
    return ParakeetEngine(quantization=quantization, model_dir=Path(path) if path else None)


def hotwords(path=None):
    """The app's ParakeetHotwordEngine (sherpa-onnx beam search + radiology hotwords); a tuned
    folder is first rebuilt in sherpa's layout by sherpa_view."""
    from sherpa_view import build
    from src.core.paths import cache_dir
    from src.core.settings import Settings
    from src.dictation.asr.engines.parakeet_hotword_engine import ParakeetHotwordEngine
    view = None
    if path:
        src = Path(path)  # experiments keep a version's weights in <version>/model, the app in fine_tuned/<version>
        view = build(src, cache_dir() / "sherpa" / (src.parent.name if src.name == "model" else src.name))
    return ParakeetHotwordEngine(Settings().get("hotword_score"), model_dir=view)


def candidates(only):
    # The app runs Parakeet at fp32; v1 and v2 exported int8 only, so int8 stock is their like-for-like row.
    # "+hw" = the same model decoded with radiology hotwords; stock-hotwords is the app's default without a voice model.
    found = [("stock-whisper", lambda: whisper(model_size="small.en")), ("stock-parakeet", parakeet),
             ("stock-parakeet-int8", lambda: parakeet(quantization="int8")), ("stock-hotwords", hotwords)]
    for v in sorted(p for p in MODELS.iterdir() if p.is_dir()):
        d = v / "model"
        if (v / "parakeet.json").exists():
            # A version moved into the app is read from the app's copy.
            if not d.exists() and v.name in available_models():
                d = model_dir_for(v.name)
            if not d.exists():
                continue
            # Versions saved before the quantization knob were int8.
            q = json.loads((v / "parakeet.json").read_text()).get("quantization", "int8")
            found.append((v.name, lambda d=d, q=q: parakeet(str(d), q)))
            if not q:  # the sherpa view is built from the full-precision files
                found.append((f"{v.name}+hw", lambda d=d: hotwords(str(d))))
        elif d.exists():
            found.append((v.name, lambda d=d: whisper(model_path=str(d))))
    unknown = sorted(set(only or ()) - {n for n, _ in found})
    if unknown:
        raise SystemExit(f"--only: no candidate named {unknown}; have {[n for n, _ in found]}")
    return [c for c in found if not only or c[0] in only]


# --features: WER drops punctuation, so it cannot see "/" or "( )". Each feature
# is a pattern; a clip counts as right when the corrected output has exactly the
# same matches as the reference, in order: same words either side of "/", same
# text inside the brackets (case and spacing ignored), field strength written
# exactly "1.5 T". Only clips whose reference has the feature are counted.
FEATURES = {
    "slash": (r"[\w.]+/[\w.]+", lambda m: m.lower()),
    "brackets": (r"\([^()]*\)", lambda m: " ".join(m.lower().split())),
    "field T": (r"\b\d+(?:\.\d+)? ?T\b", lambda m: m),
    "T1/T2": (r"\bT[12]\b", lambda m: m),
}


def feature_hits(ref, hyp):
    """{feature: True/False} for each feature present in *ref*."""
    out = {}
    for name, (pat, key) in FEATURES.items():
        want = [key(m) for m in re.findall(pat, ref)]
        if want:
            out[name] = [key(m) for m in re.findall(pat, hyp)] == want
    return out


class Score(NamedTuple):
    """One clip scored: errors and reference words, raw and after correction, then radiology terms."""
    raw_err: int
    raw_words: int
    fix_err: int
    fix_words: int
    term_err: int
    terms: int
    features: dict
    unseen: bool
    slips: int = 0
    meaning: int = 0


# label, error field, word field: each rate is summed errors over summed words, clip by clip
MEASURES = (("WER raw", "raw_err", "raw_words"), ("WER +fix", "fix_err", "fix_words"), ("term err", "term_err", "terms"),
            ("meaning", "slips", "meaning"))


def rate(recs, err, words):
    return sum(getattr(r, err) for r in recs) / max(1, sum(getattr(r, words) for r in recs))


def decode(name, make, set_name, clips, fresh):
    """[text, seconds] per clip: stored decodes, plus what is missing decoded now and stored."""
    from src.dictation.asr import TranscribeContext
    stored = {} if fresh else decode_store.load(set_name, name)
    eng = None  # loaded only if something is missing; dropped on return, so one model is in memory at a time
    for c in clips:
        key = c.audio_path.name
        if key not in stored:
            if eng is None:
                eng = make()
                eng.preload()
            t0 = time.perf_counter()
            stored[key] = [eng.transcribe(str(c.audio_path), TranscribeContext(beam_size=1)).text,
                           time.perf_counter() - t0]
    decode_store.save(set_name, name, stored)
    return [stored[c.audio_path.name] for c in clips]


def score(name, clips, decodes, lex, show):
    """Per-clip Scores and the speed (decode seconds per audio minute; nan if any clip was decoded on Kaggle)."""
    from src.dictation.postprocess.pipeline import postprocess_transcript
    recs, secs = [], [s for _, s in decodes]
    for i, (c, (text, _)) in enumerate(zip(clips, decodes)):
        fixed = postprocess_transcript(text)
        if i < show:
            print(f"[{name}] {fixed}\n{'':>{len(name) + 3}}want: {c.reference}", file=sys.stderr)
        w, p = word_error_rate(c.reference, text), word_error_rate(c.reference, fixed)
        t = term_error_rate(c.reference, text, lex)
        recs.append(Score(w.substitutions + w.deletions + w.insertions, w.ref_words,
                          p.substitutions + p.deletions + p.insertions, p.ref_words,
                          t.term_errors, t.term_tokens, feature_hits(c.reference, fixed), is_unseen(i, c),
                          *meaning_slips(c.reference, fixed)))
    audio = sum(sf.info(str(c.audio_path)).duration for c in clips)  # the manifest's figure is 0.0 when unreadable
    return recs, (math.nan if None in secs else sum(secs) / (audio / 60))


def print_rates(set_name, per_clip, speed):
    # "unseen" (a voice no version trained on) only means something on the tts set,
    # where every third clip is the held-out voice; elsewhere it is printed as "-".
    print(f"{'model':<22}{'WER raw':>9}{'WER unseen':>12}{'WER +fix':>10}{'term err':>10}{'terms':>7}{'meaning':>9}{'s/min':>8}")
    for n, recs in per_clip.items():
        unseen = f"{rate([x for x in recs if x.unseen], 'raw_err', 'raw_words'):.2%}" if set_name == "tts" else "-"
        print(f"{n:<22}{rate(recs, 'raw_err', 'raw_words'):>9.2%}{unseen:>12}{rate(recs, 'fix_err', 'fix_words'):>10.2%}"
              f"{rate(recs, 'term_err', 'terms'):>10.2%}{sum(x.terms for x in recs):>7}{rate(recs, 'slips', 'meaning'):>9.2%}{speed[n]:>8.1f}")


def print_features(per_clip):
    print("exact after correction (clips right / clips with the feature)")
    print(f"{'model':<22}" + "".join(f"{f:>14}" for f in FEATURES))
    for n, recs in per_clip.items():
        cells = [[r.features[f] for r in recs if f in r.features] for f in FEATURES]
        print(f"{n:<22}" + "".join(f"{f'{sum(c)}/{len(c)} {sum(c) / len(c):.0%}' if c else '-':>14}" for c in cells))


def print_paired(base, new, names, resamples=10_000):
    """Change B minus A, with a paired bootstrap over clips: a 95% range that
    crosses zero means the change could be luck. The range is within this set
    (these sentences, this voice), not over speakers."""
    rng = random.Random(0)
    idx = range(len(base))
    print(f"{names[1]} minus {names[0]} (points; 95% range over {resamples} resamples of {len(base)} clips)")
    for label, e, n in MEASURES:
        def pts(clips, e=e, n=n):
            return 100 * rate(clips, e, n)
        diffs = sorted(pts([new[i] for i in s]) - pts([base[i] for i in s])
                       for s in ([rng.choice(idx) for _ in idx] for _ in range(resamples)))
        print(f"{label:<10}{pts(new) - pts(base):+.2f}  "
              f"({diffs[int(0.025 * resamples)]:+.2f} to {diffs[int(0.975 * resamples)]:+.2f})")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="tts")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--paired", nargs=2, metavar=("A", "B"), help="B minus A, with a 95%% range over clips")
    ap.add_argument("--features", action="store_true", help="exact slash/bracket/field-strength/T1-T2 accuracy after correction")
    ap.add_argument("--show", type=int, default=0, help="print the first N clips' corrected output per model")
    ap.add_argument("--by-source", action="store_true",
                    help="also report each clip source separately (protocol: 'hand' = wording no model trained on)")
    ap.add_argument("--fresh", action="store_true", help="decode again instead of reusing stored decodes")
    return ap.parse_args()


def main():
    from src.medical.medical_dict import get_correction_targets
    a = parse_args()
    clips = load_set(a.set)[: a.limit or None]
    lex = list(get_correction_targets())
    per_clip, speed = {}, {}
    for name, make in candidates(a.only):
        per_clip[name], speed[name] = score(name, clips, decode(name, make, a.set, clips, a.fresh), lex, a.show)
        print(f"done {name}", file=sys.stderr, flush=True)
    groups = {"all": list(range(len(clips)))}
    if a.by_source:
        for i, c in enumerate(clips):
            groups.setdefault(c.source or "?", []).append(i)
    for g, idx in groups.items():
        print(f"\nset={a.set}  clips={len(idx)}" + (f"  source={g}" if g != "all" else "")
              + "  (lower is better; speed = seconds per audio minute)")
        sub = {n: [recs[i] for i in idx] for n, recs in per_clip.items()}
        print_rates(a.set, sub, speed)
        if a.features:
            print_features(sub)
        if a.paired:
            print_paired(sub[a.paired[0]], sub[a.paired[1]], a.paired)


if __name__ == "__main__":
    main()
