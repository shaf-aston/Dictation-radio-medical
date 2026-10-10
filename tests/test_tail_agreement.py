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
    # A third decode disagreeing even earlier must NOT shrink the screen again,
    # nor swap words without adding any.
    a.update("A lung")
    assert a.update("A lung") == "The lungs"


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


def test_a_longer_confirmed_reading_corrects_the_screen():
    """Refusing it froze the preview: one mis-heard word ("was" for "is")
    blocked every later word until the chunk closed, 3.8s measured."""
    a = LocalAgreement2()
    a.update("There was a")
    assert a.update("There was a spiculated") == "There was a"
    a.update("There is a spiculated nodule in the right")
    assert a.update("There is a spiculated nodule in the right upper") == (
        "There is a spiculated nodule in the right"
    )


def test_a_contradicting_reading_no_longer_than_the_screen_is_refused():
    """Swapping words without adding any would only make the line flicker."""
    a = LocalAgreement2()
    a.update("The lungs are clear")
    assert a.update("The lungs are clear without") == "The lungs are clear"
    a.update("The lung is clear")
    assert a.update("The lung is clear") == "The lungs are clear"


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


def test_punctuation_and_case_do_not_count_as_disagreement():
    """A growing clip re-punctuates its last word ("contrast" -> "contrast.")
    and capitalises the next sentence; the words themselves still agree."""
    a = LocalAgreement2()
    a.update("with contrast there is")
    assert a.update("with contrast. There is a nodule") == "with contrast. There is"


def test_a_shown_word_may_change_its_punctuation_but_not_its_word():
    a = LocalAgreement2()
    a.update("the lungs are")
    a.update("the lungs are clear")
    assert a.update("The lungs are clear. No") == "The lungs are clear."
    assert agreeing_prefix("lungs clear", "lungs clean") == "lungs"


def test_instant_shows_the_newest_words_before_they_are_confirmed():
    """Waiting for a second decode put every word on screen a decode late."""
    a = LocalAgreement2(show_unconfirmed=True)
    assert a.update("There is a") == "There is a"
    assert a.update("There is a spiculated nodule") == "There is a spiculated nodule"


def test_instant_lets_only_unconfirmed_words_change():
    a = LocalAgreement2(show_unconfirmed=True)
    a.update("There is a")
    a.update("There is a spiculated nodule")
    # "spiculated" is now confirmed by two decodes; "nodule" was not, so the
    # newest decode may still change it.
    assert a.update("There is a spiculated module in") == "There is a spiculated module in"


def test_instant_never_gets_shorter():
    a = LocalAgreement2(show_unconfirmed=True)
    a.update("There is a")
    a.update("There is a spiculated nodule")
    assert a.update("There is a") == "There is a spiculated nodule"
    assert a.stable() == "There is a spiculated nodule"


def test_confirmed_mode_is_the_default_class_behaviour():
    """The class alone is plain LocalAgreement-2; LiveSession opts in."""
    a = LocalAgreement2()
    a.update("There is a")
    assert a.update("There is a spiculated nodule") == "There is a"
