"""Per-accent correction patterns and accent-suggestion heuristics."""

from __future__ import annotations

from src.features.accent_corrections import (
    ACCENT_LABELS,
    ACCENT_PROFILES,
    apply_accent_corrections,
    get_available_accents,
    suggest_accent,
)


class TestAccentApplication:
    """apply_accent_corrections should fire only for the requested profile."""

    def test_neutral_is_a_no_op(self) -> None:
        assert apply_accent_corrections("wertebra", "neutral") == "wertebra"

    def test_unknown_accent_is_treated_as_neutral(self) -> None:
        assert apply_accent_corrections("wertebra", "klingon") == "wertebra"

    def test_south_asian_corrects_v_to_w_swap(self) -> None:
        assert apply_accent_corrections("wertebra", "south_asian") == "vertebra"

    def test_south_asian_corrects_th_dropping(self) -> None:
        assert apply_accent_corrections("toracic spine", "south_asian") == "thoracic spine"

    def test_middle_eastern_corrects_p_to_b_swap(self) -> None:
        assert apply_accent_corrections("bneumothorax", "middle_eastern") == "pneumothorax"

    def test_east_asian_corrects_l_to_r_swap(self) -> None:
        assert apply_accent_corrections("rigament tear", "east_asian") == "ligament tear"

    def test_west_african_corrects_th_dropping(self) -> None:
        assert apply_accent_corrections("tickness of cortex", "west_african") == "thickness of cortex"

    def test_quick_scan_short_circuit_returns_input_unchanged(self) -> None:
        text = "completely unrelated text with no accent markers"
        assert apply_accent_corrections(text, "south_asian") == text


class TestAccentSuggestion:
    """suggest_accent should pick the dominant signature or return None."""

    def test_returns_none_below_threshold(self) -> None:
        assert suggest_accent("wertebra", min_matches=2) is None

    def test_detects_south_asian_with_multiple_signatures(self) -> None:
        text = "wertebra and tandon and toracic spine"
        assert suggest_accent(text) == "south_asian"

    def test_detects_middle_eastern_signatures(self) -> None:
        text = "bosterior view shows broximal balmar findings"
        assert suggest_accent(text) == "middle_eastern"

    def test_returns_none_for_empty_input(self) -> None:
        assert suggest_accent("") is None


class TestRegistry:
    """Registry exposes the accents used by the UI."""

    def test_all_profiles_have_labels(self) -> None:
        assert set(ACCENT_PROFILES.keys()) == set(ACCENT_LABELS.keys())

    def test_get_available_accents_includes_neutral(self) -> None:
        assert "neutral" in get_available_accents()
