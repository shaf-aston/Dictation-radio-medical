"""The radiology priming vocabulary, the same for every engine.

Loaded once from src/dictation/resources/radiology_prompt.txt. It belongs to
the ASR seam, not to one engine's wrapper: anything that needs it imports it
from ``src.dictation.asr``.
"""

from __future__ import annotations

from src.features.file_manager import radiology_prompt_path


def _load_radiology_prompt() -> str:
    """Drop blank/comment lines and collapse to one line.

    Whisper truncates from the front of the prompt, so order matters: file
    content is preserved verbatim except for whitespace normalisation.
    """
    raw = radiology_prompt_path().read_text(encoding="utf-8")
    lines = [ln.strip() for ln in raw.splitlines()]
    return " ".join(ln for ln in lines if ln and not ln.startswith("#"))


RADIOLOGY_PROMPT = _load_radiology_prompt()
