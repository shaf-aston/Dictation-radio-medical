"""LocalAgreement-2: stable live display of the still-open chunk.

The open tail is re-decoded every cycle (it might still grow, or its silence
boundary just hasn't arrived yet), so its text is inherently provisional.
Showing the raw decode of a growing window makes the UI flicker as words get
revised cycle to cycle. LocalAgreement-2 instead only shows the longest
common word-prefix that agreed across the last two consecutive decodes of the
*same* open region, standard streaming-ASR practice, so the visible text is
stable even though the underlying decode is not.

Agreed text is never taken back. The prefix this returns only ever grows while
one open region is being decoded, because a preview that shrinks is worse than
one that lags: the radiologist watches words they have already read disappear
mid-sentence, which reads as the app losing their dictation.

**With one exception, and it is the first thing you see.** Waiting for two
decodes to agree means the earliest possible words are the SECOND decode, and a
decode costs over a second whatever it is handed. Measured end to end in the
browser, that put the first words of a dictation 13.8 seconds after the button
was pressed, which is long enough to believe the app is broken. So the first
decode of an open region is shown straight away, marked provisional, and the
next decode is allowed to revise it once. After that revision the never-shrink
rule holds exactly as before.

The trade is deliberate: one early correction of a word or two, in exchange for
seeing the sentence while you are still saying it. Words vanishing repeatedly
mid-dictation is the failure this class exists to prevent, and one bounded
revision at the very start is not that.

Pinning a word costs nothing, because the preview is not the report: when the
chunk closes, the ledger's own decode replaces this text wholesale and
:meth:`LocalAgreement2.reset` clears the prefix for the next open region.

Committed (ledger) text is never touched by this: only the open tail, and
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
    never withdrawn: a later decode that disagrees earlier than the last one
    did leaves the shown text alone rather than truncating it.
    """

    def __init__(self) -> None:
        self._previous: str = ""
        self._stable: str = ""
        #: True while ``_stable`` is the first decode of this open region,
        #: shown before anything has confirmed it. Exactly one revision is
        #: allowed while this holds.
        self._provisional: bool = False

    def update(self, current_tail_text: str) -> str:
        """Return the word-prefix to show for this open region, so far."""
        if not current_tail_text:
            # The caller is saying there is nothing open to preview (silence,
            # or no open chunk at all). Holding words for audio that no longer
            # has any would show a phantom tail after the committed text.
            self.reset()
            return ""

        if not self._previous:
            # Nothing to agree with yet. Show it anyway: waiting for a second
            # decode is what put the first words of a dictation thirteen
            # seconds after the button was pressed.
            self._previous = current_tail_text
            self._stable = current_tail_text
            self._provisional = True
            return self._stable

        agreed = agreeing_prefix(self._previous, current_tail_text)
        self._previous = current_tail_text

        if self._provisional:
            # The one allowed correction: what two decodes agree on replaces
            # what a single decode guessed, even where that is shorter.
            self._stable = agreed
            self._provisional = False
            return self._stable

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

    def stable(self) -> str:
        """What is currently on screen, without feeding a new decode in.

        For a cycle that decided not to decode at all: the preview should hold
        what it has, not be cleared as if the audio had gone away.
        """
        return self._stable

    def reset(self) -> None:
        """Discard agreement state: call this when the open tail's start
        moves (a chunk just closed), since the two decodes being compared
        would otherwise cover different audio and agreement would be
        meaningless. The confirmed prefix goes with it: the committed text
        now covers that audio."""
        self._previous = ""
        self._stable = ""
        self._provisional = False
