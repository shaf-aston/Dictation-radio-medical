"""LocalAgreement-2: stable live display of the still-open chunk.

The open tail is re-decoded every cycle (it might still grow, or its silence
boundary just hasn't arrived yet), so its text is inherently provisional.
Showing the raw decode of a growing window makes the UI flicker as words get
revised cycle to cycle. LocalAgreement-2 instead only shows the longest
common word-prefix that agreed across the last two consecutive decodes of the
*same* open region — standard streaming-ASR practice — so the visible text is
stable even though the underlying decode is not.

Committed (ledger) text is never touched by this — only the open tail, and
only for display. It carries no confidence and is never fed to the ledger.
"""

from __future__ import annotations


def agreeing_prefix(previous: str, current: str) -> str:
    """Longest common word-prefix of two consecutive tail decodes."""
    agreed = []
    for a, b in zip(previous.split(), current.split()):
        if a != b:
            break
        agreed.append(a)
    return " ".join(agreed)


class LocalAgreement2:
    """Feed each cycle's open-tail decode in; get the stable prefix out."""

    def __init__(self) -> None:
        self._previous: str = ""

    def update(self, current_tail_text: str) -> str:
        """Return the word-prefix that agreed between this decode and the last."""
        stable = agreeing_prefix(self._previous, current_tail_text)
        self._previous = current_tail_text
        return stable

    def reset(self) -> None:
        """Discard agreement state — call this when the open tail's start
        moves (a chunk just closed), since the two decodes being compared
        would otherwise cover different audio and agreement would be
        meaningless."""
        self._previous = ""
