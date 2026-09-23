"""Stage 7 fuzzy correction: the narrow 4-letter exception.

_MIN_LEN (5) still blocks a general fuzzy edit below that length -- a single
edit too easily lands on an unrelated real word. The one exception is exactly
4 letters, and only when the word is confirmed non-English AND exactly one
curated-lexicon term sits one edit away (see medical_dict_match._four_letter_decision).
"""

import pytest

from src.dictation.postprocess.medical_dict_match import apply_medical_dictionary_suggestions
from src.medical.medical_dict import is_english_word

pytestmark = pytest.mark.skipif(
    is_english_word("erect") is None, reason="pyspellchecker not installed: English guard unavailable"
)


def test_four_letter_typo_corrects_to_its_one_lexicon_neighbour():
    # "erct" -> "erect": exactly one radiology_lexicon.txt term at edit
    # distance 1, and pyspellchecker confirms "erct" isn't real English.
    assert apply_medical_dictionary_suggestions("erct") == "erect"


def test_four_letter_typo_in_a_hyphenated_compound_still_corrects():
    assert apply_medical_dictionary_suggestions("semi-erct") == "semi-erect"


def test_four_letter_real_english_word_is_left_alone():
    # "atom" is valid English (the guard's job) and not in any medical
    # wordlist, so it must never be treated as a typo.
    assert apply_medical_dictionary_suggestions("atom") == "atom"


def test_four_letter_word_with_two_equidistant_candidates_is_left_alone():
    # "pubi" sits one edit from both "pubic" and "pubis": an ambiguous typo
    # must not be guessed at, exactly the case _MIN_LEN=5 exists to avoid.
    assert apply_medical_dictionary_suggestions("pubi") == "pubi"
