"""The release gate: what it asks before a report leaves, and what it writes down.

Every exit (Copy, Save, Word, in both front-ends) runs these rules, so they are
tested once here rather than per button. The audit log is redirected to a
temporary file so each test can read back exactly what was recorded.
"""

import json

import pytest

from src.features import audit_log
from src.features.report_release import (
    OutstandingFindings,
    check_release,
    record_release,
)


@pytest.fixture
def audit_file(tmp_path, monkeypatch):
    path = tmp_path / "audit.log"
    monkeypatch.setattr(audit_log, "_log_path", path)
    return path


def _entries(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_clear_report_needs_no_acknowledgement():
    assert not check_release("Normal study. No pneumothorax.").needs_acknowledgement


def test_finding_needs_acknowledgement_and_names_its_level():
    check = check_release("Large right pneumothorax.")
    assert check.needs_acknowledgement
    assert check.worst_level == 1
    assert "PNEUMOTHORAX" in check.summary


def test_scanner_fault_fails_open(monkeypatch):
    # A broken scanner must never stop a report leaving.
    import src.features.report_release as rr

    def boom(_text):
        raise RuntimeError("scanner broke")

    monkeypatch.setattr(rr, "scan_for_critical_findings", boom)
    assert not check_release("Large right pneumothorax.").needs_acknowledgement


def test_acknowledgement_is_audited_per_finding(audit_file):
    check = check_release("Tension pneumothorax. Acute appendicitis.")
    record_release(check.findings, "P1", acknowledged=True)
    actions = [(e["action"], e["term"].lower()) for e in _entries(audit_file)]
    assert actions == [
        ("critical_finding_acknowledged", "tension pneumothorax"),
        ("critical_finding_acknowledged", "acute appendicitis"),
    ]


def test_override_is_audited_once(audit_file):
    check = check_release("Tension pneumothorax. Acute appendicitis.")
    record_release(check.findings, "P1", acknowledged=False)
    (entry,) = _entries(audit_file)
    assert entry["action"] == "critical_finding_overridden"


def test_answered_finding_stays_answered_and_new_one_is_outstanding():
    state = OutstandingFindings()
    state.update("Large right pneumothorax.")
    state.acknowledge()
    assert state.state == "acknowledged"
    state.update("Large right pneumothorax. Acute appendicitis.")
    assert [f.term.lower() for f in state.outstanding] == ["acute appendicitis"]


def test_reset_forgets_the_previous_patient():
    state = OutstandingFindings()
    state.update("Large right pneumothorax.")
    state.acknowledge()
    state.reset()
    state.update("Large right pneumothorax.")
    assert state.state == "outstanding"
