"""Stage 0: strip Whisper hallucinations that survived segment filtering.

Segment-level filtering happens in :mod:`src.dictation.asr.engines.faster_whisper_engine`; this stage
catches phrases that span segment boundaries or appear only after the
transcript is reassembled.
"""

from __future__ import annotations

import re
from typing import List, Pattern

_HALLUCINATION_PATTERNS: List[Pattern] = [
    # A whole line and nothing else, like the other two: unanchored at the end
    # it ate the next two words, so "Thank you\nno acute fracture" lost its
    # "no acute" and became "Fracture". [^\S\n] keeps every match on one line.
    re.compile(
        r"^thank[^\S\n]+you(?:[^\S\n]+(?:for|so))?(?:[^\S\n]+\w+){0,2}[.!]?[^\S\n]*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r"[^\S\n]*(?:see[^\S\n]+you[^\S\n]+(?:next[^\S\n]+time|in[^\S\n]+the[^\S\n]+next))[.!]?[^\S\n]*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r"^(?:subtitles?[^\S\n]+by|translated[^\S\n]+by)(?:[^\S\n]+\w+){0,2}[.!]?[^\S\n]*$",
        re.IGNORECASE | re.MULTILINE,
    ),
]

_REPETITION_LINE = re.compile(
    r"^(\b\w+\b)(?:[^\S\n]+\1){4,}[^\S\n]*[.!?]?[^\S\n]*$", re.IGNORECASE | re.MULTILINE
)


def filter_hallucinations(text: str) -> str:
    """Remove common Whisper hallucination text from the transcript."""
    for pat in _HALLUCINATION_PATTERNS:
        text = pat.sub("\n", text)
    text = _REPETITION_LINE.sub("", text)
    return text.strip()
