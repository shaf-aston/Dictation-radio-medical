"""Settings load/save persistence and convenience helpers."""

from __future__ import annotations

from src.core.settings import _DEFAULTS, Settings


class TestDefaults:
    """Defaults should be applied for any missing key."""

    def test_fresh_settings_expose_all_defaults(self) -> None:
        s = Settings()
        for key, value in _DEFAULTS.items():
            assert s.get(key) == value


class TestPersistence:
    """set/save/load should round-trip through disk."""

    def test_set_persists_value_to_new_instance(self) -> None:
        s = Settings()
        s.set("model_size", "large-v3")

        s2 = Settings()
        assert s2.get("model_size") == "large-v3"

    def test_batch_set_writes_multiple_keys_at_once(self) -> None:
        s = Settings()
        s.batch_set({"theme": "light", "font_size": 18})

        s2 = Settings()
        assert s2.get("theme") == "light"
        assert s2.get("font_size") == 18


class TestRecentReports:
    """add_recent_report should dedupe, prepend, and cap at 20 entries."""

    def test_add_dedupes_and_moves_to_front(self) -> None:
        s = Settings()
        s.add_recent_report("a.txt")
        s.add_recent_report("b.txt")
        s.add_recent_report("a.txt")

        recent = s._data["recent_reports"]
        assert recent[0] == "a.txt"
        assert recent.count("a.txt") == 1

    def test_recent_list_is_capped(self) -> None:
        s = Settings()
        for i in range(25):
            s.add_recent_report(f"r{i}.txt")
        assert len(s._data["recent_reports"]) == 20

    def test_get_recent_reports_filters_missing_files(self, tmp_path) -> None:
        existing = tmp_path / "real.txt"
        existing.write_text("x")

        s = Settings()
        s.add_recent_report(str(existing))
        s.add_recent_report(str(tmp_path / "ghost.txt"))

        assert s.get_recent_reports() == [str(existing)]
