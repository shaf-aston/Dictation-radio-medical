"""Synthesise the training set: python data/build_train.py [count] [out folder]

Voices: every installed Windows voice except settings.json's held_out_voice (so the test set has one
voice the model never heard), at three speaking rates, plus a share (`kokoro_share`) read by Kokoro
voices (`kokoro_voices`: US and UK, men and women; Apache 2.0, runs on this PC). Two Windows voices
alone left our radiology lines the narrowest part of the training mix.
A line whose audio already exists is kept, so a stopped run resumes.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pyttsx3  # noqa: E402
import soundfile as sf  # noqa: E402
from config import SETTINGS  # noqa: E402
from gen_text import generate, to_target  # noqa: E402

HELD_OUT = SETTINGS["held_out_voice"]
KOKORO = SETTINGS["kokoro_voices"]


class Kokoro:
    def __init__(self):
        from kokoro import KPipeline
        self.pipes = {lang: KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M") for lang in "ab"}

    def save(self, text: str, voice: str, wav: Path) -> None:
        audio = [r.audio.numpy() for r in self.pipes[voice[0]](text, voice=voice)]
        sf.write(wav, np.concatenate(audio), 24000)


def main(count: int, out: Path) -> None:
    out.mkdir(exist_ok=True)
    eng = pyttsx3.init()
    windows = [v.id for v in eng.getProperty("voices") if HELD_OUT not in v.id.lower()]
    base = eng.getProperty("rate")
    kokoro = Kokoro() if SETTINGS["kokoro_share"] else None
    every = round(1 / (1 - SETTINGS["kokoro_share"])) if SETTINGS["kokoro_share"] < 1 else 0
    rows = []
    for i, text in enumerate(generate(count)):
        wav = out / f"t_{i:04d}.wav"
        use_windows = every and i % every == 0
        voice = windows[(i // max(every, 1)) % len(windows)] if use_windows else KOKORO[i % len(KOKORO)]
        assert HELD_OUT not in voice.lower(), "held-out voice must never read training audio"
        if not (wav.exists() and wav.stat().st_size >= 1024):
            if use_windows:
                eng.setProperty("voice", voice)
                eng.setProperty("rate", base + (i % 3 - 1) * 20)
                eng.save_to_file(text, str(wav))
                eng.runAndWait()
            else:
                kokoro.save(text, voice, wav)
        if not wav.exists() or wav.stat().st_size < 1024:
            raise SystemExit(f"no audio for line {i}")
        rows.append({"audio": wav.name, "text": to_target(text), "spoken": text, "voice": voice.rsplit("\\", 1)[-1]})
        if i % 50 == 0:
            print(i, flush=True)
    (out / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print("done", len(rows))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 900,
         Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent / "train")
