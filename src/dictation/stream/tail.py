"""LocalAgreement-2: stable live display of the still-open chunk.

The open tail is re-decoded every cycle (it might still grow, or its silence
boundary just hasn't arrived yet), so its text is inherently provisional.
Showing the raw decode of a growing window makes the UI flicker as words get
revised cycle to cycle. LocalAgreement-2 instead only shows the longest
common word-prefix that agreed across the last two consecutive decodes of the
*same* open region — standard streaming-ASR practice — so the visible text is
stable even though the underlying decode is not.

Agreed text is never taken back. The prefix this returns only ever grows while
one open region is being decoded, because a preview that shrinks is worse than
one that lags: the radiologist watches words they have already read disappear
mid-sentence, which reads as the app losing their dictation. Measured on
``data/bench_audio/chest_long.wav``, the shown word count ran 11 → 16 → 10 → 22
without this rule — three visible retractions in one sentence.

Pinning a word costs nothing, because the preview is not the report: when the
chunk closes, the ledger's own decode replaces this text wholesale and
:meth:`LocalAgreement2.reset` clears the prefix for the next open region.

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
    """Feed each cycle's open-tail decode in; get the stable prefix out.

    The prefix grows monotonically within an open region. Two consecutive
    decodes agreeing on a word is what confirms it, and a confirmed word is
    never withdrawn — a later decode that disagrees earlier than the last one
    did leaves the shown text alone rather than truncating it.
    """

    def __init__(self) -> None:
        self._previous: str = ""
        self._stable: str = ""

    def update(self, current_tail_text: str) -> str:
        """Return the confirmed word-prefix of this open region, so far."""
        if not current_tail_text:
            # The caller is saying there is nothing open to preview (silence,
            # or no open chunk at all). Holding words for audio that no longer
            # has any would show a phantom tail after the committed text.
            self.reset()
            return ""

        agreed = agreeing_prefix(self._previous, current_tail_text)
        self._previous = current_tail_text

        # Only extend, and only along the words already shown. A longer prefix
        # that disagrees with what is on screen is a revision, not an
        # extension, and the ledger's decode is what gets to make revisions.
        agreed_words, stable_words = agreed.split(), self._stable.split()
        if (
            len(agreed_words) > len(stable_words)
            and agreed_words[: len(stable_words)] == stable_words
        ):
            self._stable = agreed
        return self._stable

    def reset(self) -> None:
        """Discard agreement state — call this when the open tail's start
        moves (a chunk just closed), since the two decodes being compared
        would otherwise cover different audio and agreement would be
        meaningless. The confirmed prefix goes with it: the committed text
        now covers that audio."""
        self._previous = ""
        self._stable = ""
