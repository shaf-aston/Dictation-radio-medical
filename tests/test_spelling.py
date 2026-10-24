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


# Written out here rather than read from the module, so a pair deleted or
# misspelt there fails this table.
PAIRS = [
    ("caliber", "calibre"), ("calibers", "calibres"), ("fiber", "fibre"), ("fibers", "fibres"),
    ("center", "centre"), ("centers", "centres"), ("centered", "centred"),
    ("tumor", "tumour"), ("tumors", "tumours"), ("color", "colour"), ("colors", "colours"),
    ("behavior", "behaviour"), ("favor", "favour"), ("favoring", "favouring"),
    ("favored", "favoured"), ("favors", "favours"), ("gray", "grey"),
    ("maneuver", "manoeuvre"), ("maneuvers", "manoeuvres"),
    ("ischemia", "ischaemia"), ("ischemic", "ischaemic"), ("anemia", "anaemia"),
    ("esophagus", "oesophagus"), ("esophageal", "oesophageal"),
    ("pediatric", "paediatric"), ("orthopedic", "orthopaedic"), ("leukemia", "leukaemia"),
    ("hematoma", "haematoma"), ("hematomas", "haematomas"),
    ("hemangioma", "haemangioma"), ("hemangiomas", "haemangiomas"), ("estrogen", "oestrogen"),
    ("diarrhea", "diarrhoea"), ("dyspnea", "dyspnoea"), ("apnea", "apnoea"),
]


@pytest.mark.parametrize("us, uk", PAIRS)
def test_every_listed_word_becomes_british_and_keeps_its_case(us, uk):
    assert apply_uk_spelling(f"the {us} and {us.capitalize()} and {us.upper()}", "en-GB") == (
        f"the {uk} and {uk.capitalize()} and {uk.upper()}"
    )


@pytest.mark.parametrize("text", ["Capsize.", "capsized", "capsizes", "capsizing", "Baize.", "Sizes."])
def test_ize_inside_a_word_is_left_alone(text):
    assert apply_uk_spelling(text, "en-GB") == text


@pytest.mark.parametrize(
    "uk, us",
    [("analyse", "analyze"), ("colour", "color"), ("centre", "center"),
     ("haematoma", "hematoma"), ("oedema", "edema"), ("favouring", "favoring"), ("grey", "gray")],
)
def test_a_british_word_knows_its_us_spelling(uk, us):
    from src.medical.spelling import us_variants
    assert us in us_variants(uk)


@pytest.fixture
def locale_setting(monkeypatch):
    from src.core import settings as core_settings
    from src.medical import spelling

    def use(value):
        class _Settings:
            def get(self, key, default=None):
                return value if key == "spelling" else default
        monkeypatch.setattr(core_settings, "Settings", _Settings)
        spelling._configured_locale.cache_clear()
    yield use
    spelling._configured_locale.cache_clear()


@pytest.mark.parametrize("value, expected", [(None, "Normal calibre."), ("en-GB", "Normal calibre."),
                                             ("en-US", "Normal caliber.")])
def test_the_spelling_setting_decides_and_defaults_to_british(locale_setting, value, expected):
    locale_setting(value)
    assert apply_uk_spelling("Normal caliber.") == expected
