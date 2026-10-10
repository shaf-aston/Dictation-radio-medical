"""LocalAgreement-2: stable live display of the still-open chunk.

The open tail is re-decoded every cycle (it might still grow, or its silence
boundary just hasn't arrived yet), so its text is inherently provisional.
Showing the raw decode of a growing window makes the UI flicker as words get
revised cycle to cycle. LocalAgreement-2 instead only shows the longest
common word-prefix that agreed across the last two consecutive decodes of the
*same* open region, standard streaming-ASR practice, so the visible text is
stable even though the underlying decode is not.

Agreed text is never taken back. The prefix this returns only ever grows longer
while one open region is being decoded, because a preview that shrinks is worse
than one that lags: the radiologist watches words they have already read disappear
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

**A confirmed longer reading may correct a shown word, never shorten the line.**
Pinning words exactly as first shown froze the preview: one mis-heard word
("There was a" for "There is a") meant no later agreement could ever extend
along it, so nothing new appeared until the chunk closed. Measured on Parakeet,
that held the screen still for 3.8s while nine words were spoken, and it was
most of the slowest tenth of preview latency. The ledger then replaced the
preview wholesale anyway: the same correction, seconds later, after a frozen
screen. So a longer prefix that two decodes agree on replaces what is shown,
even where it changes a word; the line never gets shorter, which is the
failure the never-take-back rule exists for.

**The newest words are shown before they are confirmed** (``preview_mode``
"instant", the default): the confirmed words, then whatever the latest decode
heard after them. Only those trailing, unconfirmed words can change in place;
see :meth:`LocalAgreement2.update` for the measurements.

When the chunk closes, the ledger's own decode replaces this text wholesale and
:meth:`LocalAgreement2.reset` clears the prefix for the next open region.

Committed (ledger) text is never touched by this: only the open tail, and
only for display. It carries no confidence and is never fed to the ledger.
"""

from __future__ import annotations

#: The ``preview_mode`` setting. "instant": confirmed words plus the newest
#: decode's words after them. "confirmed": confirmed words only, the older,
#: stiller, slower preview.
PREVIEW_INSTANT = "instant"
PREVIEW_CONFIRMED = "confirmed"

_PUNCT = ".,;:!?\"'()"


def _same_word(a: str, b: str) -> bool:
    """One spoken word, however the decoder dressed it this time.

    A growing clip changes how its words are written, not which words they
    are: "contrast" becomes "contrast." once the next sentence starts, "there"
    becomes "There". Parakeet punctuates and capitalises as it goes, so an
    exact comparison read those as disagreements and held confirmed words off
    the screen for another decode.
    """
    return a.strip(_PUNCT).lower() == b.strip(_PUNCT).lower()


def agreeing_prefix(previous: str, current: str) -> str:
    """Longest common word-prefix of two consecutive tail decodes, written as
    the newer decode writes it (it has heard more, so its punctuation is the
    better guess)."""
    agreed = []
    for a, b in zip(previous.split(), current.split()):
        if not _same_word(a, b):
            break
        agreed.append(b)
    return " ".join(agreed)


class LocalAgreement2:
    """Feed each cycle's open-tail decode in; get the text to show out.

    The shown text never gets shorter within an open region. Two consecutive
    decodes agreeing on a word is what confirms it: a later decode that
    disagrees earlier than the last one did leaves the shown text alone rather
    than truncating it, and a longer agreed prefix replaces it.
    """

    def __init__(self, show_unconfirmed: bool = False) -> None:
        self._previous: str = ""
        self._stable: str = ""
        #: True while ``_stable`` is the first decode of this open region,
        #: shown before anything has confirmed it. Exactly one revision is
        #: allowed while this holds.
        self._provisional: bool = False
        #: Also show the newest decode's words past the confirmed ones (the
        #: "instant" preview_mode). See :meth:`update`.
        self.show_unconfirmed = show_unconfirmed
        self._shown: str = ""

    def update(self, current_tail_text: str) -> str:
        """Return the text to show for this open region, so far.

        Confirmed words only, or, with ``show_unconfirmed``, the confirmed
        words followed by whatever the newest decode heard after them. Waiting
        for a second decode to confirm every word put each one on screen a
        whole decode late, and long medical words, which Parakeet spells
        differently from one decode to the next ("super spinetis",
        "supraspinatus"), stalled agreement exactly where it mattered.
        Measured on Parakeet with 2 of 4 cores busy elsewhere: shown-word lag
        p50 2.40s -> 1.38s, p90 4.45s -> 2.64s. The trade is that the newest,
        unconfirmed words may still change in place; confirmed ones still
        only give way to a longer confirmed reading, and the line as a whole
        never gets shorter.
        """
        confirmed = self._confirm(current_tail_text)
        if not self.show_unconfirmed or not current_tail_text:
            self._shown = confirmed
            return confirmed
        heard, held = current_tail_text.split(), confirmed.split()
        shown = confirmed
        if len(heard) > len(held) and all(map(_same_word, heard, held)):
            shown = " ".join(held + heard[len(held):])
        if len(shown.split()) < len(self._shown.split()):
            shown = self._shown
        self._shown = shown
        return shown

    def _confirm(self, current_tail_text: str) -> str:
        """The confirmed word-prefix for this open region, so far."""
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

        # Only ever longer. A longer agreed prefix that differs from what is on
        # screen is two decodes confirming a better reading; refusing it froze
        # the preview until the chunk closed (see the module docstring).
        if len(agreed.split()) > len(self._stable.split()):
            self._stable = agreed
        return self._stable

    def stable(self) -> str:
        """What is currently on screen, without feeding a new decode in.

        For a cycle that decided not to decode at all: the preview should hold
        what it has, not be cleared as if the audio had gone away.
        """
        return self._shown

    def reset(self) -> None:
        """Discard agreement state: call this when the open tail's start
        moves (a chunk just closed), since the two decodes being compared
        would otherwise cover different audio and agreement would be
        meaningless. The confirmed prefix goes with it: the committed text
        now covers that audio."""
        self._previous = ""
        self._stable = ""
        self._shown = ""
        self._provisional = False
