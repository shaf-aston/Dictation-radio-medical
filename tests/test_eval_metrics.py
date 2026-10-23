"""The scoring rules every accuracy number in this project rests on.

A metric that marks a correct transcript down sends work in the wrong
direction: every number written as digits used to score as an error against a
reference that spells it out, so engines that format numbers properly looked
worse than ones that do not.
"""

import pytest

from scripts.eval.metrics import normalize_words, word_error_rate


@pytest.mark.parametrize(
    "reference, hypothesis",
    [
        ("twelve millimetres", "12 mm"),
        ("measures fourteen by four mm", "measures 14 by 4 mm"),
        ("at c five c six", "at C5-C6"),
        ("at c five c six", "at c5c6"),
        ("l five s one", "L5/S1"),
        ("twenty five", "25"),
    ],
)
def test_spoken_and_written_forms_score_the_same(reference, hypothesis):
    assert word_error_rate(reference, hypothesis).wer == 0.0


def test_spelling_variants_still_count():
    # Deliberate: silently Americanising a British report is a real change.
    assert word_error_rate("calibre", "caliber").wer == 1.0


def test_ordinary_letters_are_not_joined_to_numbers():
    assert normalize_words("grade a 2") == ["grade", "a", "2"]
