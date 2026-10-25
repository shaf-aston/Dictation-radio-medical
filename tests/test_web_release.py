"""The web app's release gate, over HTTP: what it refuses, and what it records.

Every way a report leaves the browser (Copy, Save TXT, Word) posts here, so
the order of the questions and the audit trail are checked at the endpoint
rather than per button. The audit log goes to a temporary file.
"""

import json

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")

from src.features import audit_log  # noqa: E402
from src.ui import web_app  # noqa: E402
from src.ui.web import report, state  # noqa: E402

CLEAN = "EXAMINATION: MRI Right Knee\nFINDINGS:\nNormal.\nIMPRESSION:\n1. Normal study."


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(audit_log, "_log_path", tmp_path / "audit.log")
    state._findings.clear()
    return fastapi_testclient.TestClient(web_app.app, base_url="http://127.0.0.1:8005")


def _audit(tmp_path):
    path = tmp_path / "audit.log"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _check(client, text, **answers):
    return client.post("/api/report/check", json={"text": text, **answers})


def test_clean_report_leaves_without_a_question(client):
    assert _check(client, CLEAN).status_code == 200


def test_report_checks_are_asked_first_and_can_be_answered(client):
    text = CLEAN.replace("Normal.\n", "Marrow oedema in the left femur.\n")
    refused = _check(client, text, patient={"sex": "F"})
    assert refused.status_code == 409
    detail = refused.json()["detail"]
    assert detail["reason"] == "report_checks"
    assert any("Exam is RIGHT" in issue for issue in detail["issues"])
    assert _check(client, text, proceed_with_warnings=True).status_code == 200


def test_sex_mismatch_uses_the_patient_details_sent(client):
    text = CLEAN.replace("Normal.\n", "The prostate is normal.\n")
    assert _check(client, text, patient={"sex": "F"}).status_code == 409
    assert _check(client, text, patient={"sex": "M"}).status_code == 200


def test_unfilled_field_is_a_report_check(client):
    detail = _check(client, CLEAN + "\n[Right / Left]").json()["detail"]
    assert detail["fields"] == ["Right / Left"]


LEVEL_1 = CLEAN.replace("Normal.\n", "Large right pneumothorax.\n")
COMM = {"recipient": "Dr Patel", "method": "telephone", "at": "2026-10-23T14:20",
        "radiologist": "Dr Shah"}


def test_critical_finding_is_asked_after_the_report_checks(client):
    detail = _check(client, LEVEL_1).json()["detail"]
    assert detail["reason"] == "critical_findings"
    assert detail["worst_level"] == 1


def test_communication_sentence_is_built_by_the_server(client, monkeypatch):
    monkeypatch.setattr(state.settings(), "set", lambda *a, **k: None)
    got = client.post("/api/report/communication", json={"text": LEVEL_1, "communication": COMM}).json()
    assert got["statement"] == (
        "Finding of pneumothorax was communicated to Dr Patel by telephone "
        "at 14:20 on 23/10/2026 by Dr Shah."
    )
    assert got["text"].endswith(got["statement"] + "\n")


def test_acknowledgement_audits_who_how_and_when(client, tmp_path):
    ok = _check(client, LEVEL_1, acknowledged=True, communication=COMM)
    assert ok.status_code == 200
    ack = [e for e in _audit(tmp_path) if e["action"] == "critical_finding_acknowledged"]
    assert [(e["recipient"], e["method"], e["communicated_at"]) for e in ack] == [
        ("Dr Patel", "telephone", "2026-10-23T14:20")
    ]


def _acks(tmp_path):
    return [(e["patient_id"], e["term"]) for e in _audit(tmp_path)
            if e["action"] == "critical_finding_acknowledged"]


def test_an_answer_belongs_to_its_own_report(client, tmp_path):
    # Two patients, the same finding, one after the other (a reload, a second
    # tab or a new report): each is asked, and each call is audited.
    for report_id, patient in (("r1", "A"), ("r2", "B")):
        sent = {"report_id": report_id, "patient": {"id": patient}}
        assert _check(client, LEVEL_1, **sent).status_code == 409
        assert _check(client, LEVEL_1, acknowledged=True, communication=COMM, **sent).status_code == 200
    assert _acks(tmp_path) == [("A", "pneumothorax"), ("B", "pneumothorax")]


def test_only_findings_still_outstanding_are_asked_and_written(client, tmp_path, monkeypatch):
    monkeypatch.setattr(state.settings(), "set", lambda *a, **k: None)
    sent = {"report_id": "r1", "patient": {"id": "A"}}
    _check(client, LEVEL_1, acknowledged=True, communication=COMM, **sent)
    assert _check(client, LEVEL_1, **sent).status_code == 200  # nothing new: no question
    more = LEVEL_1.replace("IMPRESSION:", "Pneumoperitoneum.\nIMPRESSION:")
    assert _check(client, more, **sent).json()["detail"]["reason"] == "critical_findings"
    statement = client.post("/api/report/communication", json={
        "text": more, "report_id": "r1", "communication": COMM}).json()["statement"]
    assert "pneumothorax" not in statement and statement
    assert _check(client, more, acknowledged=True, communication=COMM, **sent).status_code == 200
    assert [t for _, t in _acks(tmp_path)][0] == "pneumothorax" and len(_acks(tmp_path)) == 2


def test_every_exit_is_audited(client, tmp_path):
    client.post("/api/report/check", json={"text": CLEAN, "patient": {"id": "P1"}})
    client.post("/api/report/save-txt", json={"text": CLEAN, "patient": {"id": "P1"}})
    if report.DOCX_AVAILABLE:
        client.post("/api/report/export-word", json={"text": CLEAN, "patient": {"id": "P1"}})
    actions = {e["action"] for e in _audit(tmp_path)}
    expected = {"report_copied", "report_saved"} | (
        {"report_exported"} if report.DOCX_AVAILABLE else set()
    )
    assert expected <= actions


# -- only this machine's own page is answered (DNS rebinding, cross-site socket) --

@pytest.mark.parametrize("headers, status", [
    ({}, 200),
    ({"host": "localhost:8005"}, 200),
    ({"origin": "http://127.0.0.1:8005"}, 200),
    ({"host": "evil.example:8005"}, 403),               # rebinding: the page's own name
    ({"origin": "https://evil.example"}, 403),          # another site's page
    ({"origin": "null"}, 403),
])
def test_only_this_machines_own_page_is_answered(client, headers, status):
    assert client.get("/api/runs", headers=headers).status_code == status


def test_another_sites_page_cannot_open_the_dictation_socket(client):
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("ws://127.0.0.1:8005/ws/dictate", headers={"origin": "https://evil.example"}) as ws:
            ws.receive_json()


@pytest.mark.parametrize("pid", ["李12345", "x" * 250, "../../etc/passwd", 'a"b\r\nc'])
def test_any_patient_id_exports(client, pid):
    response = client.post("/api/report/save-txt", json={"text": CLEAN, "patient": {"id": pid}})
    assert response.status_code == 200
    name = response.headers["content-disposition"].split('filename="')[1].rstrip('"')
    assert name.isascii() and "/" not in name and len(name) < 100


def test_a_request_naming_no_report_remembers_nothing(client, tmp_path):
    _check(client, LEVEL_1, acknowledged=True, communication=COMM, patient={"id": "A"})
    assert _check(client, LEVEL_1, patient={"id": "B"}).status_code == 409


def test_reloading_phrases_answers_in_the_preferences_shape(client, tmp_path, monkeypatch):
    from src.core import paths
    monkeypatch.setattr(paths, "macros_file", lambda: tmp_path / "missing.json")
    got = client.post("/api/macros/reload")  # no file: the built-in phrases, not a 500
    assert got.status_code == 200 and got.json()["macros"]["regions"]


def test_page_data_cannot_end_the_inline_script(client, monkeypatch):
    from src.ui.web import pages
    name = '<!--<script></script>&'
    monkeypatch.setattr(pages, "bootstrap_payload", lambda: {"name": name})
    html = client.get("/").text
    start = html.index("window.__BOOTSTRAP__=") + len("window.__BOOTSTRAP__=")
    injected = html[start:html.index(";", start)]
    assert "<" not in injected and ">" not in injected and "&" not in injected
    assert json.loads(injected) == {"name": name}


def test_both_front_ends_show_the_servers_words(client):
    checks = _check(client, CLEAN + "\n[Right / Left]").json()["detail"]
    assert checks["message"].endswith("Release the report anyway?")
    assert "1 unfilled template field(s)" in checks["message"]
    assert _check(client, LEVEL_1).json()["detail"]["title"] == "Life-threatening finding"
    strip = client.post("/api/report/findings", json={"text": LEVEL_1, "report_id": "r"}).json()
    assert strip["label"] == "1 finding to communicate"
    assert client.post("/api/report/findings", json={"text": CLEAN}).json()["label"] == "No findings"


def test_the_marks_hint_retires_after_the_set_number_of_uses():
    from src.medical.term_lookup import lookup_hint_active, note_lookup_used

    class Store(dict):
        def set(self, key, value):
            self[key] = value

    settings = Store(term_lookup_uses=0, term_lookup_hint_uses=2)
    assert lookup_hint_active(settings.get)
    assert note_lookup_used(settings) is True
    assert note_lookup_used(settings) is False
    assert note_lookup_used(settings) is False and settings["term_lookup_uses"] == 2


def test_the_web_patient_form_has_exactly_the_schemas_fields():
    import re

    from src.core.patient_schema import PATIENT_KEYS
    from src.ui.web.pages import frontend_file
    keys = tuple(re.findall(r'data-patient="([a-z_]+)"', frontend_file("app.html")))
    assert keys == PATIENT_KEYS
