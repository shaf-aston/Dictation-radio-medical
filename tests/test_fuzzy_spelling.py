"""Regression net for the edit-distance medical-spelling corrector (stage 7).

The fuzzy stage used a flat 94% character ratio, which silently left every
one-letter medical misspelling uncorrected ("vertabra", "atelactasis"). It now
accepts a correction only when the word is a *non-word* (per the offline English
dictionary) AND within a length-scaled edit distance of a real medical term.

These tests exercise the real on-disk wordlist, so they skip when it is absent
(it would otherwise trigger a network download). The over-correction guards
require the optional ``pyspellchecker`` dependency and skip without it.
"""

from __future__ import annotations

import pytest

import src.medical.medical_dict as md
from src.dictation.postprocess.medical_dict_match import (
    _english_known,  # type: ignore
    apply_medical_dictionary_suggestions as fix,
)
from src.features.file_manager import medical_wordlist_path


def _has_local_wordlist() -> bool:
    path = medical_wordlist_path()
    return path.exists() and path.stat().st_size > 0


pytestmark = pytest.mark.skipif(
    not _has_local_wordlist(),
    reason="bundled medical wordlist not present (would require network download)",
)

_HAS_ENGLISH_GUARD = _english_known("there") is not None


@pytest.fixture(autouse=True)
def _use_real_wordlist() -> None:
    """conftest pins the dict cache to an empty set; clear it for the real load."""
    md._TERMS = None
    md._COMMON_TERMS = None
    md._FULL_TERMS_LIST = None
    md._CORRECTION_TARGETS = None
    import src.dictation.postprocess.medical_dict_match as mdm
    mdm._MEDICAL_TERMS_CACHE = set()


# Simple one-/two-edit medical misspellings the flat ratio used to miss.
_TYPOS = {
    "vertabra": "vertebra",
    "atelactasis": "atelectasis",
    "spondilosis": "spondylosis",
    "osteophite": "osteophyte",
    "lymphadenapathy": "lymphadenopathy",
    "effussion": "effusion",
    "consolidaton": "consolidation",
    "calcificaton": "calcification",
    "degenrative": "degenerative",
    "cardiomegally": "cardiomegaly",
    "emphysima": "emphysema",
    # Radiology terms the generic wordlist does not contain (e.g. "honeycombing"
    # is absent), so before the curated lexicon these had no correct target and
    # were left misspelled. The lexicon is what now supplies the snap target.
    "honeycoming": "honeycombing",
    "bronchiectesis": "bronchiectasis",
    "diverticulitus": "diverticulitis",
}

# Words that must survive untouched: ordinary English the corrector must never
# demote to a near medical-wordlist fragment.
_ENGLISH = [
    "there", "their", "around", "again", "those", "being", "during", "since",
    "where", "further", "should", "through", "within", "otherwise", "number",
]

# Correctly-spelled clinical terms that must pass through unchanged.
_CORRECT = ["vertebra", "effusion", "pleural", "consolidation", "atelectasis"]


@pytest.mark.skipif(not _HAS_ENGLISH_GUARD, reason="pyspellchecker not installed")
@pytest.mark.parametrize("wrong,right", sorted(_TYPOS.items()))
def test_one_edit_medical_typo_is_corrected(wrong: str, right: str) -> None:
    assert fix(wrong) == right


@pytest.mark.skipif(not _HAS_ENGLISH_GUARD, reason="pyspellchecker not installed")
@pytest.mark.parametrize("word", _ENGLISH)
def test_ordinary_english_is_never_touched(word: str) -> None:
    assert fix(word) == word


@pytest.mark.parametrize("word", _CORRECT)
def test_correct_terms_pass_through(word: str) -> None:
    assert fix(word) == word
