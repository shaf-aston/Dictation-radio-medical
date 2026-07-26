"""Tests for context-aware real-word correction (stage 7.5) and split-compound
rejoin (stage 6.5).

Two properties matter, and they pull in opposite directions:

* **Recall** — confusable real words in a clear context ARE corrected
  ("spinal chord" -> "spinal cord"). This is the whole point of the stage.
* **Precision** — already-correct text is NEVER changed. Over-correcting a
  right word into a wrong medical term is the dangerous failure, so the
  precision tests are the load-bearing ones.
"""

from __future__ import annotations

import pytest

from src.dictation.postprocess.context_correct import (
    apply_context_correction,
    rejoin_split_compounds,
    _load_config,
)
from src.dictation.postprocess import context_model
from src.medical import medical_dict


@pytest.fixture(autouse=True)
def _seed_terms_for_rejoin():
    """Seed the medical-term membership set the rejoin stage needs.

    The shared conftest fixture zeroes ``medical_dict._TERMS`` to an *empty set*
    (not ``None``), so ``get_medical_terms()`` returns empty without reloading
    the real 98k wordlist — which would make the split-compound rejoin a no-op
    and the tests order-dependent. Seeding just the compounds under test keeps
    these tests fast and self-contained. Runs after the conftest fixture, so
    this binding wins.
    """
    medical_dict._TERMS = {  # type: ignore[assignment]
        "hydronephrosis", "retroperitoneal", "lymphadenopathy",
        "splenomegaly", "hepatosplenomegaly",
    }
    yield
    medical_dict._TERMS = set()  # type: ignore[assignment]


# --- Recall: real-word substitutions in clear context get fixed --------------

@pytest.mark.parametrize("heard, expected", [
    ("The spinal chord is normal.", "The spinal cord is normal."),
    ("There is know acute fracture.", "There is no acute fracture."),
    ("Course trabeculation of the bone.", "Coarse trabeculation of the bone."),
    ("A discreet nodule is present.", "A discrete nodule is present."),
    ("The perineal tendon is intact.", "The peroneal tendon is intact."),
    ("The legion demonstrates enhancement.", "The lesion demonstrates enhancement."),
])
def test_confusable_realword_corrected_in_context(heard, expected):
    assert apply_context_correction(heard) == expected


# --- Precision: correct text is left untouched (the safety guarantee) --------

@pytest.mark.parametrize("correct", [
    "The artery follows a normal course.",
    "There is coarse trabeculation of the bone.",
    "The terminal ileum is normal in calibre.",
    "There is a lucent lesion within the ilium.",
    "The peroneal tendon is intact.",
    "There is no perineal collection.",
    "A discrete nodule is present in the lung.",
    "The spinal cord is normal in signal.",
    "The vocal cords are symmetric.",
    "There is no acute fracture.",
    "There is a small pleural effusion and mild atelectasis.",
    "Of course the findings are correlated clinically.",
    # Adversarial over-correction cases (from the architect review). A generic
    # word that sits near the WRONG member must never flip a correct word.
    "There is a small lucency in the ilium.",     # "small" must not drag ilium->ileum
    "The peroneal nerve is intact.",              # peroneal nerve is valid
    "The perineal region is unremarkable.",       # perineal region is valid
    "There is coarse echotexture and normal vessels.",  # "normal" must not flip coarse->course
    "The clinician should know the findings.",    # "know" used correctly
    "The patient did not know the diagnosis.",
    "I know the warning signs were absent.",      # "signs" must not flip know->no
    "There is a small bowel lesion.",             # generic words must not touch "lesion"
])
def test_correct_text_is_not_over_corrected(correct):
    assert apply_context_correction(correct) == correct


def test_only_switches_within_a_set_never_to_outside_word():
    # "cord" can only ever become "chord" (its set-mate), never anything else.
    _, _ = _load_config()
    out = apply_context_correction("The spinal cord appears normal.")
    assert out == "The spinal cord appears normal."


# --- Casing is preserved on correction ---------------------------------------

def test_casing_preserved():
    assert apply_context_correction("Chord signal is normal in the spinal region.") \
        .startswith("Cord")


def test_never_correct_word_switched_without_a_cue():
    # "chord" is virtually never correct in radiology (never-correct tier), so
    # it is fixed even with no cue word nearby...
    assert apply_context_correction("Chord signal abnormality is present.") \
        == "Cord signal abnormality is present."
    # ...but "know" (which CAN be correct) is left alone without positive
    # evidence — the prior alone must not flip it.
    assert apply_context_correction("The patient did not know the results.") \
        == "The patient did not know the results."


# --- Split-compound rejoin ----------------------------------------------------

@pytest.mark.parametrize("heard, expected", [
    ("hydro nephrosis", "hydronephrosis"),
    ("retro peritoneal", "retroperitoneal"),
    ("lymph adenopathy", "lymphadenopathy"),
    ("hepato spleno megaly", "hepatosplenomegaly"),  # three-way, fixed-point
])
def test_split_compound_rejoined(heard, expected):
    assert rejoin_split_compounds(heard) == expected


@pytest.mark.parametrize("text", [
    "There is no evidence of fracture.",   # two plain words, never welded
    "The bowel wall is normal.",
    "Of course this is normal.",
])
def test_rejoin_does_not_weld_ordinary_words(text):
    assert rejoin_split_compounds(text) == text


# --- Regression: the YAML boolean trap ---------------------------------------

def test_no_member_is_not_the_boolean_false():
    """`- members: [no, know]` unquoted parses `no` as Python False, which
    would silently make the member "false" and corrupt "no acute" -> "false
    acute". Guard: the "no"/"know" set must correct toward the real word "no"."""
    assert apply_context_correction("There is know acute injury.") \
        == "There is no acute injury."
    # And "no" must never be rewritten to the literal "false".
    assert "false" not in apply_context_correction("There is no acute injury.").lower()


# --- Context model sanity -----------------------------------------------------

def test_context_model_prefers_seen_collocation():
    context_model.reset_cache()
    m = context_model.get_context_model()
    # "spinal cord" is in the seed corpus; "spinal chord" is not — so "cord"
    # should score higher than "chord" after the word "spinal".
    assert m.score("spinal", "cord", "is") > m.score("spinal", "chord", "is")
