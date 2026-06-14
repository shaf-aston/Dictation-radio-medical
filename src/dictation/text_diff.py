"""Boundary-overlap detection shared between worker and tests.

The live worker emits a sliding window of audio every cycle.  Adjacent
windows overlap by :data:`OVERLAP_SEC` seconds, which means the same
words can appear at the end of the committed prefix and the start of
the new chunk text.  This module finds and trims that duplication.
"""

from __future__ import annotations

import difflib
import re
import string
from typing import Optional, Tuple

# Public — re-used by callers that want the same tokenisation.
WORD_TOKEN_RE = re.compile(r"\S+|\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _strip(word: str) -> str:
    """Lower-case + punctuation-stripped form used purely for matching."""
    return word.lower().translate(_PUNCT_TABLE)


def find_overlap(
    committed_tail: list[str],
    chunk_head: list[str],
    min_size: int = 2,
) -> Optional[Tuple[int, int]]:
    """Return ``(tail_match_start, chunk_match_start_with_size)`` or ``None``.

    Compares the lower-cased, punctuation-stripped words.  The match
    must either start at index 0 of ``chunk_head`` or reach the end of
    ``committed_tail`` — otherwise it's not a true boundary overlap.
    """
    tail = [_strip(w) for w in committed_tail]
    head = [_strip(w) for w in chunk_head]

    matcher = difflib.SequenceMatcher(None, tail, head)
    m = matcher.find_longest_match(0, len(tail), 0, len(head))

    if m.size < min_size:
        return None
    return None if m.b != 0 and m.a + m.size != len(tail) else (m.a, m.b + m.size)


def trim_committed_tail(committed: str, chunk_text: str, lookback: int = 20) -> str:
    """Concatenate ``committed`` with the new portion of ``chunk_text``.

    If a boundary overlap exists (last words of ``committed`` repeated at
    the start of ``chunk_text``), the duplicated portion of ``chunk_text``
    is removed before concatenation.  When no overlap is found and the
    tail of ``committed`` is a clean prefix of ``chunk_text``, the prefix
    is trimmed defensively to avoid the classic "word word word" repeat.
    """
    if not committed:
        return chunk_text.lstrip()

    c_words = committed.split()
    chunk_tokens = WORD_TOKEN_RE.findall(chunk_text)
    head_indexed = [(i, t) for i, t in enumerate(chunk_tokens) if t.strip()]

    tail = c_words[-lookback:]
    head = [t for _, t in head_indexed[:lookback]]

    overlap = find_overlap(tail, head)
    if overlap is not None:
        _, new_word_idx = overlap
        if new_word_idx >= len(head_indexed):
            return committed
        token_idx = head_indexed[new_word_idx][0]
        new_text = "".join(chunk_tokens[token_idx:])
        sep = "" if committed.endswith((" ", "\n")) else " "
        return committed + sep + new_text.lstrip(" ")

    # No clean overlap — defensive trim if chunk begins with the exact tail
    # of committed (handles the case where punctuation differed slightly).
    chunk_lower = chunk_text.lstrip().lower()
    for k in range(min(len(c_words), lookback), 1, -1):
        candidate = " ".join(c_words[-k:]).lower()
        if chunk_lower.startswith(candidate):
            remainder = chunk_text.lstrip()[len(candidate):]
            return committed + remainder
    sep = "" if committed.endswith((" ", "\n")) else " "
    return committed + sep + chunk_text.lstrip()
