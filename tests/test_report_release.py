"""The release gate: what it asks before a report leaves, and what it writes down.

Every exit (Copy, Save, Word, in both front-ends) runs these rules, so they are
tested once here rather than per button. The audit log is redirected to a
temporary file so each test can read back exactly what was recorded.
"""

import json
from datetime import datetime

import pytest

from src.features import audit_log
from src.features.report_release import (
    Communication,
    OutstandingFindings,
    add_communication_statement,
    check_release,
    communication_statement,
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


COMM = Communication("Dr Patel", "telephone", datetime(2026, 10, 23, 14, 20), "Dr Shah")


def test_communication_statement_says_who_how_when_and_by_whom():
    check = check_release("Tension pneumothorax.")
    assert communication_statement(check.findings, COMM) == (
        "Finding of tension pneumothorax was communicated to Dr Patel "
        "by telephone at 14:20 on 23/10/2026 by Dr Shah."
    )


def test_several_findings_are_listed_in_one_sentence():
    check = check_release("Tension pneumothorax. Acute appendicitis.")
    assert communication_statement(check.findings, COMM).startswith(
        "Findings of tension pneumothorax and acute appendicitis were communicated"
    )


def test_statement_is_added_once():
    statement = communication_statement(check_release("Tension pneumothorax.").findings, COMM)
    once = add_communication_statement("Tension pneumothorax.", statement)
    assert once == "Tension pneumothorax.\n\n" + statement + "\n"
    assert add_communication_statement(once, statement) == once


def test_audit_records_recipient_method_time_and_radiologist(audit_file):
    record_release(check_release("Tension pneumothorax.").findings, "P1", True, COMM)
    (entry,) = _entries(audit_file)
    assert (entry["recipient"], entry["method"], entry["communicated_at"], entry["radiologist"]) == (
        "Dr Patel", "telephone", "2026-10-23T14:20", "Dr Shah"
    )


def test_three_findings_are_listed_with_commas_and_a_final_and():
    check = check_release("Tension pneumothorax. Acute appendicitis. Volvulus.")
    assert communication_statement(check.findings, COMM).startswith(
        "Findings of tension pneumothorax, acute appendicitis and volvulus were communicated"
    )


def test_statement_without_a_radiologist_name_ends_at_the_date():
    comm = Communication("Dr Patel", "telephone", COMM.at)
    statement = communication_statement(check_release("Tension pneumothorax.").findings, comm)
    assert statement.endswith("at 14:20 on 23/10/2026.")


def test_acknowledgement_without_details_is_audited_with_blank_fields(audit_file):
    record_release(check_release("Tension pneumothorax.").findings, "P1", True)
    (entry,) = _entries(audit_file)
    assert (entry["recipient"], entry["method"], entry["communicated_at"], entry["radiologist"]) == (
        "", "", "", ""
    )


def test_override_names_every_finding_once(audit_file):
    record_release(check_release("Tension pneumothorax. Volvulus.").findings, "P1", False)
    (entry,) = _entries(audit_file)
    assert entry["terms"] == "Tension pneumothorax; Volvulus"


def test_the_count_display_reads_clear_outstanding_then_acknowledged():
    state = OutstandingFindings()
    state.update("Normal study.")
    assert (state.state, state.count) == ("clear", 0)
    state.update("Tension pneumothorax.")
    assert state.state == "outstanding"
    state.acknowledge()
    assert state.state == "acknowledged"


def test_a_broken_consistency_check_never_blocks_a_report(monkeypatch):
    from src.features import report_checks
    from src.features.report_release import report_warnings

    def broken(*_a, **_k):
        raise RuntimeError("checker fault")
    monkeypatch.setattr(report_checks, "check_report", broken)
    found = report_warnings("MRI [Right / Left] knee.")
    assert found.issues == () and found.fields == ("Right / Left",)
