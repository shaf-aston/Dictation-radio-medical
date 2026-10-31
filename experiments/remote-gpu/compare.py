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


def candidates(only):
    # The app runs Parakeet at fp32; v1 and v2 exported int8 only, so int8 stock is their like-for-like row.
    found = [("stock-whisper", lambda: whisper(model_size="small.en")), ("stock-parakeet", parakeet),
             ("stock-parakeet-int8", lambda: parakeet(quantization="int8"))]
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
        elif d.exists():
            found.append((v.name, lambda d=d: whisper(model_path=str(d))))
    unknown = sorted(set(only or ()) - {n for n, _ in found})
    if unknown:
        raise SystemExit(f"--only: no candidate named {unknown}; have {[n for n, _ in found]}")
    return [c for c in found if not only or c[0] in only]


# --features: WER drops punctuation, so it cannot see "/" or "( )". Each feature
# is a pattern; a clip counts as right when the corrected output has exactly the
# same matches as the reference (lower-cased, in order). Only clips whose
# reference has the feature are counted.
FEATURES = {
    "slash": (r"[\w.]+/[\w.]+", lambda m: m.lower()),
    "brackets": (r"\([^()]*\)", lambda m: "()"),
    "field T": (r"\b\d+(?:\.\d+)? ?T\b", lambda m: m.replace(" ", "")),
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
    a = ap.parse_args()

    from src.dictation.asr import TranscribeContext
    from src.dictation.postprocess.pipeline import postprocess_transcript
    from src.medical.medical_dict import get_correction_targets

    clips = load_set(a.set)[: a.limit or None]
    lex = list(get_correction_targets())
    rows, per_clip, feats = [], {}, {}
    for name, make in candidates(a.only):
        eng = make()
        eng.preload()
        raw, proc, unseen_raw, term_err, term_tok, dec, audio = [], [], [], 0, 0, 0.0, 0.0
        for i, c in enumerate(clips):
            t0 = time.perf_counter()
            text = eng.transcribe(str(c.audio_path), TranscribeContext(beam_size=1)).text
            dec += time.perf_counter() - t0
            audio += sf.info(str(c.audio_path)).duration  # the manifest's figure is 0.0 when it could not be read
            raw.append((c.reference, text))
            fixed = postprocess_transcript(text)
            proc.append((c.reference, fixed))
            for f, ok in feature_hits(c.reference, fixed).items():
                feats.setdefault(name, {}).setdefault(f, []).append(ok)
            if i < a.show:
                print(f"[{name}] {fixed}\n{'':>{len(name) + 3}}want: {c.reference}", file=sys.stderr)
            if is_unseen(i, c):
                unseen_raw.append((c.reference, text))
            t = term_error_rate(c.reference, text, lex)
            term_err += t.term_errors
            term_tok += t.term_tokens
            w = word_error_rate(c.reference, text)
            per_clip.setdefault(name, []).append(
                (w.substitutions + w.deletions + w.insertions, w.ref_words, t.term_errors, t.term_tokens))
        rows.append((name, wer(raw), wer(unseen_raw), wer(proc),
                     term_err / term_tok if term_tok else 0.0, dec / (audio / 60)))
        print(f"done {name}", file=sys.stderr, flush=True)

    print(f"\nset={a.set}  clips={len(clips)}  (lower is better; speed = seconds per audio minute)")
    print(f"{'model':<22}{'WER raw':>9}{'WER unseen':>12}{'WER +fix':>10}{'term err':>10}{'s/min':>8}")
    for n, w, u, p, t, s in rows:
        print(f"{n:<22}{w:>9.2%}{u:>12.2%}{p:>10.2%}{t:>10.2%}{s:>8.1f}")
    if a.features:
        print(f"\nexact after correction (clips right / clips with the feature)")
        print(f"{'model':<22}" + "".join(f"{f:>14}" for f in FEATURES))
        for n, *_ in rows:
            cells = [feats.get(n, {}).get(f, []) for f in FEATURES]
            print(f"{n:<22}" + "".join(f"{f'{sum(c)}/{len(c)} {sum(c) / len(c):.0%}' if c else '-':>14}" for c in cells))
    if a.paired:
        print_paired(*(per_clip[n] for n in a.paired), a.paired)


def print_paired(base, new, names, resamples=10_000):
    """Change B minus A in WER and term error, with a paired bootstrap over clips:
    a 95% range that crosses zero means the change could be luck."""
    import random

    def rate(clips, e, n):
        return 100 * sum(c[e] for c in clips) / max(1, sum(c[n] for c in clips))

    rng = random.Random(0)
    idx = range(len(base))
    print(f"\n{names[1]} minus {names[0]} (points; 95% range over {resamples} resamples)")
    for label, e, n in (("WER", 0, 1), ("term err", 2, 3)):
        diffs = sorted(rate([new[i] for i in s], e, n) - rate([base[i] for i in s], e, n)
                       for s in ([rng.choice(idx) for _ in idx] for _ in range(resamples)))
        print(f"{label:<10}{rate(new, e, n) - rate(base, e, n):+.2f}  "
              f"({diffs[int(0.025 * resamples)]:+.2f} to {diffs[int(0.975 * resamples)]:+.2f})")


if __name__ == "__main__":
    main()
