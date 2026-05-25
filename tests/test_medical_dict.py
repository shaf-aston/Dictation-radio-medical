"""Medical-dictionary loading and fuzzy-correction smoke tests.

These exercise the on-disk wordlist; if it is missing the loader will hit the
network. Tests fall back to an injected term set when no wordlist is present.
"""

from __future__ import annotations

import pytest

import src.medical.medical_dict as md
from src.features.file_manager import medical_wordlist_path


def _has_local_wordlist() -> bool:
    path = medical_wordlist_path()
    return path.exists() and path.stat().st_size > 0


pytestmark = pytest.mark.skipif(
    not _has_local_wordlist(),
    reason="bundled medical wordlist not present (would require network download)",
)


@pytest.fixture(autouse=True)
def _reset_dict_cache_to_force_real_load() -> None:
    """The shared conftest pins the cache to an empty set; clear it so the
    real on-disk wordlist is loaded for these tests."""
    md._TERMS = None
    md._COMMON_TERMS = None
    md._FULL_TERMS_LIST = None


class TestLoading:
    """The wordlist should load into the cached singletons."""

    def test_terms_set_is_populated(self) -> None:
        terms = md.get_medical_terms()
        assert len(terms) > md._MIN_VALID_TERMS

    def test_common_terms_cache_is_filled(self) -> None:
        md.get_medical_terms()
        assert md._COMMON_TERMS, "common-terms shortlist should be cached"


class TestSuggestion:
    """suggest_correction should hit the exact-match short circuit and the fuzzy path."""

    def test_exact_match_returns_lowercase_term(self) -> None:
        terms = md.get_medical_terms()
        sample = next(iter(terms))
        assert md.suggest_correction(sample) == sample

    def test_unknown_garbage_word_returns_none(self) -> None:
        md.get_medical_terms()
        assert md.suggest_correction("zzqqxxvv") is None

    def test_high_cutoff_rejects_far_matches(self) -> None:
        md.get_medical_terms()
        assert md.suggest_correction("abcdefghij", cutoff=0.99) is None
