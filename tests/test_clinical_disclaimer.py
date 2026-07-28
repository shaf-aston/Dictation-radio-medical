"""The shared clinical disclaimer: the wording, and the "have they seen it" decision."""

from __future__ import annotations

from src.features.clinical_disclaimer import (
    DISCLAIMER_TEXT,
    DISCLAIMER_TITLE,
    mark_shown,
    needs_showing,
)


class FakeSettings:
    """Just the two settings methods the service uses, counting the writes."""

    def __init__(self, initial: dict | None = None) -> None:
        self._data = dict(initial or {})
        self.writes = 0

    def get(self, key, default=None):
        return self._data.get(key, default)

    def set(self, key, value) -> None:
        self._data[key] = value
        self.writes += 1


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------

def test_a_fresh_install_still_needs_the_disclaimer() -> None:
    assert needs_showing(FakeSettings())


def test_marking_it_shown_settles_it() -> None:
    settings = FakeSettings()
    mark_shown(settings)
    assert not needs_showing(settings)
    assert settings.get("disclaimer_shown") is True


def test_marking_it_shown_twice_is_harmless() -> None:
    # Both front-ends write the same flag, and the web one can be acknowledged
    # from two tabs. A second call must not error or re-write the file.
    settings = FakeSettings()
    mark_shown(settings)
    mark_shown(settings)
    assert not needs_showing(settings)
    assert settings.writes == 1


def test_an_already_acknowledged_install_is_not_asked_again() -> None:
    assert not needs_showing(FakeSettings({"disclaimer_shown": True}))


# ---------------------------------------------------------------------------
# The wording — a liability statement, not copy
# ---------------------------------------------------------------------------

def test_the_load_bearing_claims_survive_any_edit() -> None:
    # These three are the whole point of the notice: the model is not cleared
    # for clinical use, and a human must check its output. An edit that drops
    # one of them changes what the radiologist was told, so it fails here.
    assert "FDA" in DISCLAIMER_TEXT
    assert "CE-marked" in DISCLAIMER_TEXT
    assert "reviewed" in DISCLAIMER_TEXT
    assert DISCLAIMER_TITLE.strip()
