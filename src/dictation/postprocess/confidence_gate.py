"""Confidence veto — keep the guessing stages off words the decoder was sure about.

Several pipeline stages rewrite a word purely because it looks like something in
a vocabulary (SymSpell edit-distance snapping above all). They fire just as
readily on a word that was *heard correctly* as on one that was misheard, which
is what the eval harness reports as the **false-correction rate**: the share of
the pipeline's edits that took a right word and made it wrong.

The decoder already knows which words it was sure about
(:class:`src.dictation.asr.types.Word` carries a per-word confidence). This
module is the gate that uses it: run the guessing stages as before, then revert
any rewrite whose original words were all above the ceiling.

Two rules the rest of the pipeline depends on:

* ``None`` means "no confidence is known for this word" — an engine without word
  probabilities, or a word some earlier stage invented. Unknown is never treated
  as confident, so a missing signal can only make the gate fire *less*.
* Only ``replace`` edits are gated. A ``delete`` is how "scratch that" and the
  hallucination filter do their job, and an ``insert`` (punctuation) has no
  original word to be confident about.

Pure: no I/O, no model, no Qt. The caller supplies the ceiling and logs what the
gate refused.
"""

from __future__ import annotations

import difflib
import re
from typing import List, Optional, Sequence, Tuple

#: Words are whitespace-delimited runs. Matching them *with their positions*
#: (rather than rebuilding from ``str.split()``) is what lets a veto be spliced
#: back in without flattening the paragraph breaks the rest of the text carries.
_WORD = re.compile(r"\S+")


def carry_confidences(
    before_words: Sequence[str],
    after_words: Sequence[str],
    confidences: Sequence[Optional[float]],
) -> List[Optional[float]]:
    """Realign a per-word confidence list across one text edit.

    Confidences arrive aligned to the words the decoder produced, but the gate
    runs several stages later, on text those stages have already edited. This
    walks the word-level diff and moves each confidence to where its word ended
    up: ``equal`` carries it through, ``delete`` drops it, and ``replace`` /
    ``insert`` produce words no decoder ever emitted, so those get ``None``.

    Returns a list the same length as *after_words*. A confidence list that
    doesn't match *before_words* is not alignable at all, so every word comes
    back ``None`` (unknown) rather than being shifted onto the wrong words.
    """
    if len(confidences) != len(before_words):
        return [None] * len(after_words)

    out: List[Optional[float]] = []
    matcher = difflib.SequenceMatcher(
        None, list(before_words), list(after_words), autojunk=False
    )
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            out.extend(confidences[i1:i2])
        elif op != "delete":  # replace / insert -> newly written words
            out.extend([None] * (j2 - j1))
    return out


def veto_confident_rewrites(
    before: str,
    after: str,
    confidences: Optional[Sequence[Optional[float]]],
    ceiling: Optional[float],
) -> Tuple[str, List[str]]:
    """Undo the rewrites that touched only words the decoder was sure about.

    Args:
        before: Text entering the guessing stages.
        after:  Text they produced.
        confidences: One entry per word of *before*; ``None`` where unknown.
        ceiling: Confidence at or above which a word is protected.

    Returns:
        ``(text, vetoed)`` — the text with confident rewrites restored, and the
        ``'"orig" -> "repl"'`` spans that were refused. The list is the gate's
        evidence: what it blocked is surfaced to the caller, never swallowed.

    A whole ``replace`` span is restored only when *every* original word in it
    is known-confident. A span that mixes a sure word with an unsure one is let
    through: a span is restored as a unit, so vetoing a mixed span would also
    block the fix the unsure word probably needs. The gate protects words it can
    protect on their own.

    Fails safe: no confidences, no ceiling, or a list whose length doesn't match
    *before*'s words all return *after* unchanged with no vetoes — no signal
    means today's exact behaviour, never a half-applied gate.
    """
    before_words = before.split()
    if not confidences or ceiling is None or len(confidences) != len(before_words):
        return after, []

    after_spans = [(m.group(), m.start(), m.end()) for m in _WORD.finditer(after)]
    after_words = [span[0] for span in after_spans]

    edits: List[Tuple[int, int, str]] = []
    vetoed: List[str] = []
    matcher = difflib.SequenceMatcher(None, before_words, after_words, autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op != "replace":
            continue
        span = confidences[i1:i2]
        if not span or any(c is None or c < ceiling for c in span):
            continue
        orig = " ".join(before_words[i1:i2])
        edits.append((after_spans[j1][1], after_spans[j2 - 1][2], orig))
        vetoed.append(f'"{orig}" -> "{" ".join(after_words[j1:j2])}"')

    text = after
    for start, end, orig in reversed(edits):  # right-to-left keeps offsets valid
        text = text[:start] + orig + text[end:]
    return text, vetoed
