"""Stage 0: strip Whisper hallucinations that survived segment filtering.

Segment-level filtering happens in :mod:`src.dictation.asr.engines.faster_whisper_engine`; this stage
catches phrases that span segment boundaries or appear only after the
transcript is reassembled.
"""

from __future__ import annotations

import re
from typing import List, Pattern, Tuple

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


#: Every pattern this stage removes, for the live post-processor, which must
#: not split its text beside one (a removed line takes a line break with it).
REMOVED_PATTERNS: Tuple[Pattern, ...] = (*_HALLUCINATION_PATTERNS, _REPETITION_LINE)


def filter_hallucinations(text: str) -> str:
    """Remove common Whisper hallucination text from the transcript.

    A hallucinated line is removed with the line breaks that separated it from
    the text before it (or after it, when it is the first line), so the text
    reads as if it had never been said. Replacing it with a line break, as this
    used to, left blank lines where it had been.
    """
    for pat in REMOVED_PATTERNS:
        text = _drop(pat, text)
    return text.strip()


def _drop(pattern: Pattern, text: str) -> str:
    out, pos = [], 0
    for match in pattern.finditer(text):
        start, end = match.start(), match.end()
        if match.group(0).strip() and (start == 0 or text[start - 1] == "\n"):
            # A whole line: take its separating breaks with it.
            if start > 0:
                while start > pos and text[start - 1] == "\n":
                    start -= 1
            else:
                while end < len(text) and text[end] == "\n":
                    end += 1
        out.append(text[pos:start])
        pos = end
    out.append(text[pos:])
    return "".join(out)
