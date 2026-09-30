"""Speak typed text into a WAV, so dictation can be tested without a voice.

Developer-only, and removable: delete this folder plus the ``/api/dev/speak``
endpoint in ``src/ui/web_app.py`` and the "Test voice" row in the developer
console (search app.js for ``speechTest``). Nothing else depends on it.

Uses the Windows speech engine through ``pyttsx3``: free, offline, already
installed. Synthesis runs in its own process because the engine keeps
per-thread state that hangs when reused inside a long-running server.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

MAX_CHARS = 2000        # a long report, not a novel
TIMEOUT_SEC = 60


def speak_to_wav(text: str) -> bytes:
    """Return *text* spoken as WAV bytes. Raises ValueError or RuntimeError."""
    text = text.strip()
    if not text or len(text) > MAX_CHARS:
        raise ValueError(f"Text must be 1 to {MAX_CHARS} characters")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "speech.wav"
        done = subprocess.run(
            [sys.executable, "-m", "src.devtools.speech_test", str(out)],
            input=text, text=True, capture_output=True, timeout=TIMEOUT_SEC,
        )
        if done.returncode != 0 or not out.exists() or out.stat().st_size < 1024:
            raise RuntimeError(f"Speech synthesis failed: {done.stderr.strip()[-300:]}")
        return out.read_bytes()


def _synthesise(text: str, out: str) -> None:
    import pyttsx3
    engine = pyttsx3.init()
    engine.save_to_file(text, out)
    engine.runAndWait()


if __name__ == "__main__":
    _synthesise(sys.stdin.read(), sys.argv[1])
