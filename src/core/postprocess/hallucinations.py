"""Stage 0 — strip Whisper hallucinations that survived segment filtering.

Segment-level filtering happens in :mod:`src.core.transcriber`; this stage
catches phrases that span segment boundaries or appear only after the
transcript is reassembled.
"""

from __future__ import annotations

import re
from typing import List, Pattern

_HALLUCINATION_PATTERNS: List[Pattern] = [
    re.compile(
        r"^(?:thank you(?:\s+(?:for|so))?(?:\s+\w+){0,2})[.!]?\s*",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r"\s*(?:see you\s+(?:next\s+time|in\s+the\s+next))[.!]?\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r"^(?:subtitles?\s+by|translated\s+by)(?:\s+\w+){0,2}[.!]?\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
]

_REPETITION_LINE = re.compile(
    r"^(\b\w+\b)(?:\s+\1){4,}\s*[.!?]?\s*$", re.IGNORECASE | re.MULTILINE
)


def filter_hallucinations(text: str) -> str:
    """Remove common Whisper hallucination text from the transcript."""
    for pat in _HALLUCINATION_PATTERNS:
        text = pat.sub("\n", text)
    text = _REPETITION_LINE.sub("", text)
    return text.strip()
