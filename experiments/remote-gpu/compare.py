"""Stock model vs every tuned version, same audio, one table.

    python experiments/remote-gpu/compare.py [--set tts] [--only NAME ...] [--limit N]

Candidates: 'stock-whisper' (small.en), 'stock-parakeet' (the app's default engine)
plus each folder in models/ that holds a model/ directory; a folder with parakeet.json
is a Parakeet version (read from data/models/fine_tuned/ once moved into the app),
any other a Whisper one. Nothing here touches src/ or the app settings.
Scores the model's raw text (no correction stage) so the model alone is judged,
plus the text after the app's correction stage, which is what the user sees.
'unseen' = clips in a voice the model was never trained on: the tts set's held-out
voice (every 3rd clip, starting at the 2nd) or, for the own set, every clip.
Tuned voices are 'seen', so read 'unseen' first.
"""
import argparse
import json
import sys
import time

import soundfile as sf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODELS, REPO  # noqa: E402

sys.path.insert(0, str(REPO))
from scripts.eval.corpus import load_set  # noqa: E402
from scripts.eval.metrics import term_error_rate, word_error_rate  # noqa: E402
from src.dictation.asr.engines.parakeet_engine import ParakeetEngine, available_models, model_dir_for  # noqa: E402


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
    from src.dictation.asr.engines.parakeet_hotword_engine import ParakeetHotwordEngine
    from src.core.settings import Settings
    return ParakeetHotwordEngine(Settings().get("hotword_score"), model_dir=build(Path(path)) if path else None)


def decode_cache(set_name, model):
    """Raw text per clip, {audio file: [text, decode seconds or None]}; run_train.py files the kernel's here too."""
    from src.core.paths import cache_dir
    return cache_dir() / "compare_decodes" / set_name / f"{model}.json"


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
    import re
    out = {}
    for name, (pat, key) in FEATURES.items():
        want = [key(m) for m in re.findall(pat, ref)]
        if want:
            out[name] = [key(m) for m in re.findall(pat, hyp)] == want
    return out


def wer(pairs):
    """Errors summed clip by clip over all reference words. Joining the clips into one
    string would let the aligner match words across clip boundaries."""
    if not pairs:
        return float("nan")
    scores = [word_error_rate(r, h) for r, h in pairs]
    return sum(w.substitutions + w.deletions + w.insertions for w in scores) / max(1, sum(w.ref_words for w in scores))


def main():
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
    a = ap.parse_args()

    from src.dictation.asr import TranscribeContext
    from src.dictation.postprocess.pipeline import postprocess_transcript
    from src.medical.medical_dict import get_correction_targets

    clips = load_set(a.set)[: a.limit or None]
    lex = list(get_correction_targets())
    groups = {"all": list(range(len(clips)))}
    if a.by_source:
        for i, c in enumerate(clips):
            groups.setdefault(c.source or "?", []).append(i)
    # Per model, per clip: (raw errors, words, fixed errors, words, term errors, term words, features, unseen)
    per_clip, speed = {}, {}
    for name, make in candidates(a.only):
        # A version is never edited after it is saved, so its raw text per clip is decoded once and
        # kept; re-scoring (a new correction rule, a new feature) then takes seconds. --fresh redoes it.
        cache = decode_cache(a.set, name)
        stored = {} if a.fresh or not cache.exists() else json.loads(cache.read_text(encoding="utf-8"))
        eng = None
        dec, audio, recs = 0.0, 0.0, []
        for i, c in enumerate(clips):
            key = c.audio_path.name
            if key not in stored:
                if eng is None:
                    eng = make()
                    eng.preload()
                t0 = time.perf_counter()
                stored[key] = [eng.transcribe(str(c.audio_path), TranscribeContext(beam_size=1)).text,
                               time.perf_counter() - t0]
            text, secs = stored[key]
            dec = None if dec is None or secs is None else dec + secs  # None: decoded on Kaggle's GPU, no CPU timing
            audio += sf.info(str(c.audio_path)).duration  # the manifest's figure is 0.0 when it could not be read
            fixed = postprocess_transcript(text)
            if i < a.show:
                print(f"[{name}] {fixed}\n{'':>{len(name) + 3}}want: {c.reference}", file=sys.stderr)
            w, p = word_error_rate(c.reference, text), word_error_rate(c.reference, fixed)
            t = term_error_rate(c.reference, text, lex)
            recs.append((w.substitutions + w.deletions + w.insertions, w.ref_words,
                         p.substitutions + p.deletions + p.insertions, p.ref_words,
                         t.term_errors, t.term_tokens, feature_hits(c.reference, fixed), is_unseen(i, c)))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(stored), encoding="utf-8")
        per_clip[name], speed[name] = recs, (float("nan") if dec is None else dec / (audio / 60))
        print(f"done {name}", file=sys.stderr, flush=True)
        del eng  # one model in memory at a time

    # "unseen" (a voice no version trained on) only means something on the tts set,
    # where every third clip is the held-out voice; elsewhere it is printed as "-".
    show_unseen = a.set == "tts"
    for g, idx in groups.items():
        print(f"\nset={a.set}  clips={len(idx)}" + (f"  source={g}" if g != "all" else "")
              + "  (lower is better; speed = seconds per audio minute)")
        print(f"{'model':<22}{'WER raw':>9}{'WER unseen':>12}{'WER +fix':>10}{'term err':>10}{'terms':>7}{'s/min':>8}")
        for n, recs in per_clip.items():
            r = [recs[i] for i in idx]
            u = [x for x in r if x[7]]
            print(f"{n:<22}{_rate(r, 0, 1):>9.2%}{(f'{_rate(u, 0, 1):.2%}' if show_unseen else '-'):>12}"
                  f"{_rate(r, 2, 3):>10.2%}{_rate(r, 4, 5):>10.2%}{sum(x[5] for x in r):>7}{speed[n]:>8.1f}")
        if a.features:
            print("exact after correction (clips right / clips with the feature)")
            print(f"{'model':<22}" + "".join(f"{f:>14}" for f in FEATURES))
            for n, recs in per_clip.items():
                cells = [[recs[i][6][f] for i in idx if f in recs[i][6]] for f in FEATURES]
                print(f"{n:<22}" + "".join(f"{f'{sum(c)}/{len(c)} {sum(c) / len(c):.0%}' if c else '-':>14}"
                                           for c in cells))
        if a.paired:
            print_paired(*([per_clip[n][i] for i in idx] for n in a.paired), a.paired)


def _rate(recs, e, n):
    return sum(r[e] for r in recs) / max(1, sum(r[n] for r in recs))


def print_paired(base, new, names, resamples=10_000):
    """Change B minus A, with a paired bootstrap over clips: a 95% range that
    crosses zero means the change could be luck. The range is within this set
    (these sentences, this voice), not over speakers."""
    import random

    def rate(clips, e, n):
        return 100 * _rate(clips, e, n)

    rng = random.Random(0)
    idx = range(len(base))
    print(f"{names[1]} minus {names[0]} (points; 95% range over {resamples} resamples of {len(base)} clips)")
    for label, e, n in (("WER raw", 0, 1), ("WER +fix", 2, 3), ("term err", 4, 5)):
        diffs = sorted(rate([new[i] for i in s], e, n) - rate([base[i] for i in s], e, n)
                       for s in ([rng.choice(idx) for _ in idx] for _ in range(resamples)))
        print(f"{label:<10}{rate(new, e, n) - rate(base, e, n):+.2f}  "
              f"({diffs[int(0.025 * resamples)]:+.2f} to {diffs[int(0.975 * resamples)]:+.2f})")


if __name__ == "__main__":
    main()
