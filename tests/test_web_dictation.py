"""The web app's dictation socket and bootstrap, driven without a model.

The live session is replaced by a stand-in that returns fixed text, so these
tests are about what the server does with that text: here, answering a
selected pick-list field the way the desktop does.
"""

from types import SimpleNamespace

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")

from src.features import template_fields  # noqa: E402
from src.ui import web_app  # noqa: E402
from src.ui.web import dictate, pages  # noqa: E402


class _FakeSession:
    """Says *text* for everything: live, handed back and polished."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.audio_sec = 1.0
        self.chunks_decoded = 1
        self.committed_sec = 1.0
        self.uncertain_words = ()

    def feed(self, _samples) -> None:
        pass

    def cycle(self):
        return SimpleNamespace(
            committed=self.text, preview="", state="live", audio_sec=1.0, uncertain=()
        )

    def close_open_tail_fast(self) -> None:
        pass

    def committed_text(self) -> str:
        return self.text

    def finalize(self, cancelled=None) -> str:
        return self.text

    def close(self) -> None:
        pass


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(dictate, "record_web_run", lambda *a, **k: None)
    return fastapi_testclient.TestClient(web_app.app)


def _dictate(client, monkeypatch, spoken: str, field=None) -> dict:
    monkeypatch.setattr(dictate, "build_live_session", lambda *_: _FakeSession(spoken))
    with client.websocket_connect("/ws/dictate") as ws:
        if field:
            ws.send_json({"command": "field", "text": field})
        ws.send_json({"command": "stop"})
        messages = {}
        while "final" not in messages:
            msg = ws.receive_json()
            messages[msg["type"]] = msg
    return messages


def test_saying_an_option_answers_the_selected_pick_list(client, monkeypatch):
    got = _dictate(client, monkeypatch, "Left.", field="[Right / Left]")
    assert got["stopped"]["text"] == "Left"
    assert got["final"]["text"] == "Left"


def test_dictation_without_a_field_is_left_as_spoken(client, monkeypatch):
    got = _dictate(client, monkeypatch, "Left.")
    assert got["final"]["text"] == "Left."


def test_the_page_gets_the_servers_field_pattern():
    assert pages.bootstrap_payload()["field_pattern"] == template_fields.FIELD_PATTERN
