"""Stock model vs every tuned version, same audio, one table.

    python experiments/remote-gpu/compare.py [--set tts] [--only NAME ...] [--limit N]

Candidates: 'stock' (the app's normal small.en) plus each folder in models/
that holds a model/ directory. Nothing here touches src/ or the app settings.
Scores the model's raw text (no correction stage) so the model alone is judged,
plus the text after the app's correction stage, which is what the user sees.
'unseen' = clips spoken by the voice held out of training (tts set: every 3rd
clip, starting at the 2nd). Tuned voices are 'seen', so read 'unseen' first.
"""
import argparse, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODELS  # noqa: E402
from scripts.eval.corpus import load_set  # noqa: E402
from scripts.eval.metrics import term_error_rate, word_error_rate  # noqa: E402


def candidates(only):
    found = [("stock", {"model_size": "small.en"})]
    for d in sorted(MODELS.glob("*/model")):
        found.append((d.parent.name, {"model_path": str(d)}))
    return [c for c in found if not only or c[0] in only]


def wer(pairs):
    ref = " ".join(r for r, _ in pairs); hyp = " ".join(h for _, h in pairs)
    return word_error_rate(ref, hyp).as_dict()["wer"] if pairs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="tts")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from src.dictation.asr import TranscribeContext
    from src.dictation.asr.engines.faster_whisper_engine import FasterWhisperEngine
    from src.dictation.postprocess.pipeline import postprocess_transcript
    from src.medical.medical_dict import get_correction_targets

    clips = load_set(a.set)[: a.limit or None]
    lex = list(get_correction_targets())
    rows = []
    for name, kw in candidates(a.only):
        eng = FasterWhisperEngine(**kw)
        eng.preload()
        raw, proc, unseen_raw, term_err, term_tok, dec, audio = [], [], [], 0, 0, 0.0, 0.0
        for i, c in enumerate(clips):
            t0 = time.perf_counter()
            text = eng.transcribe(str(c.audio_path), TranscribeContext(beam_size=1)).text
            dec += time.perf_counter() - t0
            audio += c.duration_sec
            raw.append((c.reference, text))
            proc.append((c.reference, postprocess_transcript(text)))
            if i % 3 == 1:
                unseen_raw.append((c.reference, text))
            t = term_error_rate(c.reference, text, lex)
            term_err += t.term_errors; term_tok += t.term_tokens
        rows.append((name, wer(raw), wer(unseen_raw), wer(proc),
                     term_err / term_tok if term_tok else 0.0, dec / (audio / 60)))
        print(f"done {name}", file=sys.stderr, flush=True)

    print(f"\nset={a.set}  clips={len(clips)}  (lower is better; speed = seconds per audio minute)")
    print(f"{'model':<22}{'WER raw':>9}{'WER unseen':>12}{'WER +fix':>10}{'term err':>10}{'s/min':>8}")
    for n, w, u, p, t, s in rows:
        print(f"{n:<22}{w:>9.2%}{u:>12.2%}{p:>10.2%}{t:>10.2%}{s:>8.1f}")


if __name__ == "__main__":
    main()
