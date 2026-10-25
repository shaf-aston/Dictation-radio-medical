"""Two Settings objects on one file must not erase each other's changes.

The desktop app, the web app and several modules each hold their own Settings.
Saving used to write the whole in-memory dict, so whichever saved last put
back every key the other had changed since it loaded: a lost update that
looked like a setting "not sticking".
"""

import json

import pytest

from src.core import settings as settings_module


@pytest.fixture
def settings_path(tmp_path, monkeypatch):
    path = tmp_path / "dictation_settings.json"
    monkeypatch.setattr(settings_module, "settings_file", lambda: path)
    return path


def test_two_writers_keep_both_changes(settings_path):
    a = settings_module.Settings()
    b = settings_module.Settings()  # loaded before a writes
    a.set("radiologist_name", "Dr Shah")
    b.set("theme", "light")
    on_disk = json.loads(settings_path.read_text())
    assert on_disk["radiologist_name"] == "Dr Shah"
    assert on_disk["theme"] == "light"


def test_defaults_are_not_written_to_the_file(settings_path):
    s = settings_module.Settings()
    s.set("theme", "light")
    # A default stays a default, so a better default later reaches this install.
    assert json.loads(settings_path.read_text()) == {"theme": "light"}
    assert s.get("asr_hotwords") is True


def test_recent_reports_survive_another_writer(settings_path):
    a = settings_module.Settings()
    b = settings_module.Settings()
    a.add_recent_report("/r/one.txt")
    b.set("theme", "light")
    assert json.loads(settings_path.read_text())["recent_reports"] == ["/r/one.txt"]


def test_a_file_that_does_not_parse_is_kept_not_overwritten(tmp_path):
    from src.core.json_store import read_json, write_json
    path = tmp_path / "dictation_settings.json"
    broken = '{"radiologist_name": "Dr Shah", "web_port": 8005,}'  # one trailing comma
    path.write_text(broken)
    assert read_json(path, {}) == {}
    write_json(path, {"theme": "light"})
    (kept,) = tmp_path.glob("dictation_settings.json.corrupt-*")
    assert kept.read_text() == broken
