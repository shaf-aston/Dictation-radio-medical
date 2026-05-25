"""AdaptiveLearning: passive learning from edits, manual learn API, persistence."""

from __future__ import annotations

import src.features.adaptive_learning as al


def fresh() -> al.AdaptiveLearning:
    """Return a clean singleton; the conftest fixture resets state per test."""
    return al.get_adaptive_learning()


class TestLearnCorrection:
    """Explicit corrections should be remembered and applied via apply()."""

    def test_correction_round_trips_through_apply(self) -> None:
        engine = fresh()
        engine.learn_correction("teh", "the")

        assert engine.get_correction("teh") == "the"
        assert engine.apply_learned_corrections("teh quick fox") == "the quick fox"

    def test_too_short_correction_is_ignored(self) -> None:
        engine = fresh()
        engine.learn_correction("a", "an")
        assert engine.get_correction("a") is None

    def test_identity_correction_is_ignored(self) -> None:
        engine = fresh()
        engine.learn_correction("word", "word")
        assert engine.get_correction("word") is None


class TestCustomTerms:
    """learn_term should populate the custom vocabulary set."""

    def test_term_is_remembered(self) -> None:
        engine = fresh()
        engine.learn_term("supraspinatus")
        assert engine.is_known_term("Supraspinatus")


class TestTrackEdit:
    """Single-word edits should be picked up as corrections."""

    def test_single_word_swap_is_learned(self) -> None:
        engine = fresh()
        engine.track_edit("meniscal tier", "meniscal tear")
        assert engine.get_correction("tier") == "tear"

    def test_multi_word_edit_is_not_learned(self) -> None:
        engine = fresh()
        engine.track_edit("alpha beta gamma", "delta epsilon zeta")
        assert engine.get_correction("alpha") is None


class TestStatsAndReset:
    """Stats reflect counts and reset clears every store."""

    def test_get_stats_reports_counts(self) -> None:
        engine = fresh()
        engine.learn_correction("teh", "the")
        engine.learn_term("supraspinatus")
        stats = engine.get_stats()
        assert stats["corrections"] == 1
        assert stats["custom_terms"] == 1

    def test_reset_clears_all_stores(self) -> None:
        engine = fresh()
        engine.learn_correction("teh", "the")
        engine.learn_term("supraspinatus")

        engine.reset()

        stats = engine.get_stats()
        assert stats == {
            "corrections": 0,
            "custom_terms": 0,
            "tracked_terms": 0,
            "accent_hints": 0,
        }


class TestPersistence:
    """force_save followed by a fresh singleton should reload data."""

    def test_corrections_survive_reload(self) -> None:
        engine = fresh()
        engine.learn_correction("teh", "the")
        engine.force_save()

        # Drop singleton to force reload
        al.AdaptiveLearning._instance = None
        al._adaptive_learning = None

        reloaded = al.get_adaptive_learning()
        assert reloaded.get_correction("teh") == "the"


class TestPromptAdditions:
    """Custom prompt additions should join custom + frequent terms."""

    def test_custom_terms_appear_in_prompt(self) -> None:
        engine = fresh()
        engine.learn_term("subscapularis")
        prompt = engine.get_custom_prompt_additions()
        assert "subscapularis" in prompt
