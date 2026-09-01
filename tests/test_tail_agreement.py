"""The live preview must never take words back.

Whisper re-decodes the still-open tail every cycle and revises it freely. The
old stabiliser returned the agreement of the *last two* decodes only, so the
shown text shrank whenever a decode disagreed earlier than the one before it —
measured as 11 -> 16 -> 10 -> 22 words on data/bench_audio/chest_long.wav.
Words vanishing mid-sentence reads as the app losing the dictation.
"""

import pytest

from src.dictation.stream.tail import LocalAgreement2, agreeing_prefix


def test_nothing_is_shown_until_two_decodes_agree():
    """A single decode is a guess; agreement is what confirms a word."""
    a = LocalAgreement2()
    assert a.update("The lungs are clear") == ""
    assert a.update("The lungs are clear without") == "The lungs are clear"


def test_a_later_disagreement_cannot_shrink_what_is_on_screen():
    """The exact regression: the prefix held instead of truncating."""
    a = LocalAgreement2()
    a.update("The lungs are clear")
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
    assert a.update("Something else entirely") == ""  # starts confirming afresh


def test_a_closed_chunk_resets_the_preview():
    """The committed decode now covers that audio — the preview must not repeat it."""
    a = LocalAgreement2()
    a.update("The lungs are clear")
    a.update("The lungs are clear")
    a.reset()
    assert a.update("No pneumothorax") == ""


def test_agreeing_prefix_stops_at_the_first_differing_word():
    assert agreeing_prefix("a b c", "a b d") == "a b"
    assert agreeing_prefix("", "a b") == ""
    assert agreeing_prefix("a b", "") == ""


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
