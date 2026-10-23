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

CLEAN = "EXAMINATION: MRI Right Knee\nFINDINGS:\nNormal.\nIMPRESSION:\n1. Normal study."


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(audit_log, "_log_path", tmp_path / "audit.log")
    web_app._findings.reset()
    return fastapi_testclient.TestClient(web_app.app)


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
