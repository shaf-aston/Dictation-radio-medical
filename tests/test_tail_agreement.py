"""The live preview shows the first decode at once, then never takes words back.

Whisper re-decodes the still-open tail every cycle and revises it freely. The
old stabiliser returned the agreement of the *last two* decodes only, so the
shown text shrank whenever a decode disagreed earlier than the one before it.
Words vanishing mid-sentence reads as the app losing the dictation.

Waiting for that agreement, though, means the earliest words you can possibly
see are the SECOND decode, and a decode costs over a second whatever it is
handed: measured end to end, the first words of a dictation landed 13.8 seconds
after the button was pressed. So the first decode is shown immediately and may
be revised exactly once. These tests pin both halves: that the first words
appear straight away, and that the revision is bounded to one.
"""

import pytest

from src.dictation.stream.tail import LocalAgreement2, agreeing_prefix


def test_the_first_decode_is_shown_at_once():
    """Waiting for a second decode is what made the first words take 13.8s."""
    a = LocalAgreement2()
    assert a.update("The lungs are clear") == "The lungs are clear"


def test_the_first_decode_may_be_corrected_exactly_once():
    """The provisional guess gives way to what two decodes agree on..."""
    a = LocalAgreement2()
    assert a.update("The lungs are clear") == "The lungs are clear"
    # The second decode disagrees from the third word on. The shown text is
    # cut back to the agreement -- the one revision the preview is allowed.
    assert a.update("The lungs were clearly") == "The lungs"


def test_and_never_again_after_that():
    """...and from then on the never-shrink rule holds exactly as before."""
    a = LocalAgreement2()
    a.update("The lungs are clear")
    assert a.update("The lungs were clearly") == "The lungs"
    # A third decode disagreeing even earlier must NOT shrink the screen again.
    a.update("A lung was clear")
    assert a.update("A lung was clear") == "The lungs"


def test_a_later_disagreement_cannot_shrink_what_is_on_screen():
    """The exact regression: the prefix held instead of truncating."""
    a = LocalAgreement2()
    a.update("The lungs are clear")
    # Spends the one allowed revision; the two decodes agree in full here.
    assert a.update("The lungs are clear without focal") == "The lungs are clear"
    # Whisper now revises 'lungs' -> 'lung'. Agreement of the last two decodes
    # is just "The", which is what used to be displayed.
    assert a.update("The lung is clear without focal") == "The lungs are clear"


def test_agreement_still_extends_along_the_words_already_shown():
    a = LocalAgreement2()
    a.update("The lungs are clear")
    assert a.update("The lungs are clear without") == "The lungs are clear"
    assert a.update("The lungs are clear without focal") == "The lungs are clear without"


def test_a_longer_prefix_that_contradicts_the_screen_is_refused():
    """Longer is not automatically better: it must agree with what was shown."""
    a = LocalAgreement2()
    a.update("The lungs are clear")
    assert a.update("The lungs are clear without") == "The lungs are clear"
    # Both decodes agree on five words, but the second word differs from the
    # text already read. That is a revision, and only the ledger revises.
    a.update("The lung is clear without focal")
    assert a.update("The lung is clear without focal") == "The lungs are clear"


def test_an_empty_decode_clears_the_preview():
    """Silence, or no open chunk: holding words would show a phantom tail."""
    a = LocalAgreement2()
    a.update("The lungs are clear")
    a.update("The lungs are clear")
    assert a.update("") == ""
    # The next region starts fresh, so its own first decode shows at once.
    assert a.update("Something else entirely") == "Something else entirely"


def test_a_closed_chunk_resets_the_preview():
    """The committed decode now covers that audio, so the old prefix goes and
    the new region is provisional again, first decode shown at once."""
    a = LocalAgreement2()
    a.update("The lungs are clear")
    a.update("The lungs are clear")
    a.reset()
    assert a.update("No pneumothorax") == "No pneumothorax"
    assert a.update("No pneumothorax is seen") == "No pneumothorax"


def test_agreeing_prefix_stops_at_the_first_differing_word():
    assert agreeing_prefix("a b c", "a b d") == "a b"
    assert agreeing_prefix("", "a b") == ""
    assert agreeing_prefix("a b", "") == ""


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
