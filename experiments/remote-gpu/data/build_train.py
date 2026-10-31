"""Synthesise the training set: python data/build_train.py [count]

Voices: every installed voice except settings.json's held_out_voice, so the test
set has one voice the model never heard. Three speaking rates per voice.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pyttsx3  # noqa: E402
from config import SETTINGS  # noqa: E402
from gen_text import generate, to_target  # noqa: E402

OUT = Path(__file__).resolve().parent / "train"
HELD_OUT = SETTINGS["held_out_voice"]


def main(count: int) -> None:
    OUT.mkdir(exist_ok=True)
    eng = pyttsx3.init()
    voices = [v.id for v in eng.getProperty("voices") if HELD_OUT not in v.id.lower()]
    base = eng.getProperty("rate")
    rows = []
    for i, text in enumerate(generate(count)):
        wav = OUT / f"t_{i:04d}.wav"
        eng.setProperty("voice", voices[i % len(voices)])
        eng.setProperty("rate", base + (i % 3 - 1) * 20)
        eng.save_to_file(text, str(wav))
        eng.runAndWait()
        if not wav.exists() or wav.stat().st_size < 1024:
            raise SystemExit(f"no audio for line {i}")
        rows.append({"audio": wav.name, "text": to_target(text), "spoken": text})
        if i % 50 == 0:
            print(i, flush=True)
    (OUT / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print("done", len(rows))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 900)
