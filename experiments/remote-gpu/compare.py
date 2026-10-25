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
    a = ap.parse_args()

    from src.dictation.asr import TranscribeContext
    from src.dictation.postprocess.pipeline import postprocess_transcript
    from src.medical.medical_dict import get_correction_targets

    clips = load_set(a.set)[: a.limit or None]
    lex = list(get_correction_targets())
    rows = []
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
            proc.append((c.reference, postprocess_transcript(text)))
            if is_unseen(i, c):
                unseen_raw.append((c.reference, text))
            t = term_error_rate(c.reference, text, lex)
            term_err += t.term_errors
            term_tok += t.term_tokens
        rows.append((name, wer(raw), wer(unseen_raw), wer(proc),
                     term_err / term_tok if term_tok else 0.0, dec / (audio / 60)))
        print(f"done {name}", file=sys.stderr, flush=True)

    print(f"\nset={a.set}  clips={len(clips)}  (lower is better; speed = seconds per audio minute)")
    print(f"{'model':<22}{'WER raw':>9}{'WER unseen':>12}{'WER +fix':>10}{'term err':>10}{'s/min':>8}")
    for n, w, u, p, t, s in rows:
        print(f"{n:<22}{w:>9.2%}{u:>12.2%}{p:>10.2%}{t:>10.2%}{s:>8.1f}")


if __name__ == "__main__":
    main()
