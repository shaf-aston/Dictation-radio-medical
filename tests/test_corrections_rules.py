"""The real-word homophones that no other stage can catch.

A mishear that lands on a valid English word is invisible to the spell
checker and to the fuzzy matcher, so it reaches the radiologist looking
confident and unmarked.  "hyla" (a genus of tree frog) is the observed
case: it silently replaced "hila" in a chest report on 2026-09-19.
Only a named rule in corrections.yaml catches this class.
"""

from src.dictation.postprocess.pipeline import postprocess_transcript
from src.dictation.postprocess.rules import apply_data_rules


def test_hyla_becomes_hila():
    assert "hila" in apply_data_rules("the hyla are unremarkable in contour")
    assert "hyla" not in apply_data_rules("the hyla are unremarkable in contour")


def test_hylar_becomes_hilar():
    assert apply_data_rules("hylar prominence") == "hilar prominence"


def test_sentence_capitalisation_survives_the_fix():
    # The rule substitutes a lowercase literal; the capitalisation stage
    # further down the pipeline is what puts the sentence back together.
    assert postprocess_transcript("hyla and hylar structures are normal") == (
        "Hila and hilar structures are normal"
    )


def test_already_correct_text_is_left_alone():
    # Idempotence: running the fix on a correct report must change nothing.
    correct = "The hila are unremarkable in contour"
    assert postprocess_transcript("the hila are unremarkable in contour") == correct
    assert apply_data_rules(correct) == correct


def test_longer_real_words_containing_hyla_are_untouched():
    # The nearest dangerous case the bug report does not name: "hylan" is a
    # real injectable and "hylae" a real plural.  A rule without word
    # boundaries would corrupt both.
    assert apply_data_rules("hylan G-F 20 injection") == "hylan G-F 20 injection"
    assert apply_data_rules("hylae") == "hylae"
