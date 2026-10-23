"""British spelling of ordinary words, and the words it must leave alone."""

import pytest

from src.medical.medical_dict import is_english_word
from src.medical.spelling import apply_uk_spelling


@pytest.mark.parametrize(
    "us, uk",
    [
        ("Normal caliber.", "Normal calibre."),
        ("Difficult to characterize.", "Difficult to characterise."),
        ("Findings favoring infection.", "Findings favouring infection."),
        ("Gray matter.", "Grey matter."),
        ("Localized collection, generalized change.", "Localised collection, generalised change."),
        ("Analyzed.", "Analysed."),
        ("CHARACTERIZED", "CHARACTERISED"),
    ],
)
def test_us_spelling_becomes_british(us, uk):
    assert apply_uk_spelling(us, "en-GB") == uk


@pytest.mark.parametrize("text", ["Lesion size 5 mm.", "Seize activity.", "A prize.", "Ultrasound."])
def test_words_where_ize_is_not_a_suffix_are_left_alone(text):
    assert apply_uk_spelling(text, "en-GB") == text


def test_us_locale_changes_nothing():
    assert apply_uk_spelling("Normal caliber, characterized.", "en-US") == "Normal caliber, characterized."


@pytest.mark.parametrize(
    "word", ["characterise", "favouring", "calibre", "localised", "colour", "centre",
             "haematoma", "oesophagus", "paediatric", "behaviour"],
)
def test_british_spelling_is_a_real_word(word):
    # Without this the fuzzy corrector rewrote "characterise" to "characters".
    assert is_english_word(word) is True


def test_a_real_typo_is_still_not_a_word():
    assert is_english_word("atelactasis") is False


def test_fuzzy_stage_leaves_british_spelling_alone():
    from src.dictation.postprocess.medical_dict_match import apply_medical_dictionary_suggestions
    text = "Difficult to characterise. Localised collection."
    assert apply_medical_dictionary_suggestions(text) == text
