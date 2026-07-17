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
    _SYMSPELL_AVAILABLE,  # type: ignore
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
    # SymSpell is keyed by (len(terms), len(lexicon)); clearing terms without
    # clearing the SymSpell instance leaves the stale index active.
    md._SYMSPELL = None
    md._SYMSPELL_SIGNATURE = None
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
    # Correct only in the generic ~98k wordlist, not the 756-term curated
    # lexicon — these were uncorrectable when the snap-target pool was
    # restricted to the lexicon alone (SymSpell now searches the full
    # membership set; see medical_dict_match.py module docstring).
    "esophogeal": "esophageal",
    "gastroesophogeal": "gastroesophageal",
    "thryoid": "thyroid",
}

# Words that must survive untouched: ordinary English the corrector must never
# demote to a near medical-wordlist fragment.
_ENGLISH = [
    "there", "their", "around", "again", "those", "being", "during", "since",
    "where", "further", "should", "through", "within", "otherwise", "number",
]

# Correctly-spelled clinical terms that must pass through unchanged.
_CORRECT = ["vertebra", "effusion", "pleural", "consolidation", "atelectasis"]

# Rare-but-correct terms absent from both wordlists, with no candidate within
# the accepted edit distance — must not be pulled toward an unrelated term now
# that the candidate pool is ~98k terms instead of 756.
_RARE_UNKNOWN = ["polyradiculopathy", "spondylodiscitis"]

# Correctly-spelled British medical terms whose American counterpart
# ("haematology" -> "hematology") is the only one of the pair in the ~98k
# wordlist. Widening the SymSpell candidate pool to that wordlist must not
# silently rewrite these to American spelling.
_UK_SPELLINGS = [
    "oesophageal", "oesophagus", "anaesthetic", "anaesthesia", "leukaemia",
    "diarrhoea", "foetus", "foetal", "haemoglobin", "caesarean", "gynaecology",
    "gynaecological", "oestrogen", "haemodynamic", "haemothorax", "haemoptysis",
    "faecal", "faeces", "ischaemia", "ischaemic", "oedematous", "anaemic",
    "haemorrhagic", "haematological", "haematology", "haemostasis",
]


@pytest.mark.skipif(
    not _HAS_ENGLISH_GUARD or not _SYMSPELL_AVAILABLE,
    reason="pyspellchecker or symspellpy not installed — edit-distance corrections unavailable",
)
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


@pytest.mark.skipif(not _HAS_ENGLISH_GUARD, reason="pyspellchecker not installed")
@pytest.mark.parametrize("word", _RARE_UNKNOWN)
def test_rare_correct_terms_not_pulled_to_unrelated_word(word: str) -> None:
    assert fix(word) == word


@pytest.mark.skipif(not _HAS_ENGLISH_GUARD, reason="pyspellchecker not installed")
@pytest.mark.parametrize("word", _UK_SPELLINGS)
def test_british_spelling_not_corrected_to_american(word: str) -> None:
    assert fix(word) == word
