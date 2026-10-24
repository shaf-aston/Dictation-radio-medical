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
    state.findings.reset()
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
