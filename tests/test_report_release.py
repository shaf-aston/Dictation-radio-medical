"""The shared report-release gate: the two decisions, the audit trail, and the
guarantee that every desktop exit consults them."""

from __future__ import annotations

import ast
from pathlib import Path

from src.features import report_release
from src.features.report_release import (
    OutstandingFindings, ReleaseCheck, check_release, record_release, unfilled_fields,
)
from src.medical.critical_findings import CriticalFinding

_URGENT_TEXT = "Findings: Large right pneumothorax with mediastinal shift."


# ---------------------------------------------------------------------------
# unfilled_fields — the cancellable decision
# ---------------------------------------------------------------------------

def test_a_filled_in_report_has_no_unfilled_fields() -> None:
    assert unfilled_fields("Findings: No acute cardiopulmonary process.") == ()


def test_one_placeholder_is_returned_without_its_brackets() -> None:
    assert unfilled_fields("Findings: [FINDINGS]") == ("FINDINGS",)


def test_both_placeholder_styles_are_found_in_first_appearance_order() -> None:
    text = "Impression: [IMPRESSION]\nPatient: {{patient_name}}\nBody: [FINDINGS]"
    assert unfilled_fields(text) == ("IMPRESSION", "patient_name", "FINDINGS")


def test_the_same_field_left_in_twice_is_reported_once() -> None:
    # The question is which fields are unfilled, not how many brackets there are:
    # a repeat would inflate the count the radiologist is shown.
    assert unfilled_fields("[FINDINGS] ... more ... [FINDINGS]") == ("FINDINGS",)


def test_ordinary_bracketed_text_is_not_a_placeholder() -> None:
    # Checked against the regex, not assumed: a placeholder is SHOUTED and at
    # least two characters, so measurements and sentence-case asides are text.
    text = "Nodule [5 mm] in [Segment] six, per [A] prior study."
    assert unfilled_fields(text) == ()


# ---------------------------------------------------------------------------
# check_release — the decision
# ---------------------------------------------------------------------------

def test_an_ordinary_report_needs_no_acknowledgement() -> None:
    check = check_release("Findings: No acute cardiopulmonary process.")
    assert not check.needs_acknowledgement
    assert check.summary == ""
    assert check.findings == ()


def test_whitespace_only_text_is_clear() -> None:
    assert not check_release("   \n\t ").needs_acknowledgement


def test_an_urgent_finding_needs_acknowledgement_and_is_summarised() -> None:
    check = check_release(_URGENT_TEXT)
    assert check.needs_acknowledgement
    assert "pneumothorax" in check.summary.lower()
    assert [f.term.lower() for f in check.findings] == ["pneumothorax"]
    assert check.worst_level == 1


def test_worst_level_is_the_most_severe_finding_not_the_last_one() -> None:
    # Level 1 outranks level 2. Reporting the wrong end of the range would
    # downgrade a life-threatening finding to "urgent" in the dialog, which
    # picks the icon the radiologist sees.
    check = check_release(
        "Findings: Acute appendicitis. There is also a large pneumothorax."
    )
    levels = {f.level for f in check.findings}
    assert levels == {1, 2}, f"expected both severities, got {levels}"
    assert check.worst_level == 1


def test_a_scanner_fault_is_treated_as_clear_and_logged(monkeypatch, caplog) -> None:
    # Fail open on purpose: a crashing scanner must not stop a radiologist
    # sending a report. The failure is logged, not swallowed silently.
    def _boom(_text):
        raise RuntimeError("scanner exploded")

    monkeypatch.setattr(report_release, "scan_for_critical_findings", _boom)
    with caplog.at_level("WARNING"):
        check = check_release(_URGENT_TEXT)

    assert not check.needs_acknowledgement
    assert "scanner exploded" in caplog.text


# ---------------------------------------------------------------------------
# OutstandingFindings — acknowledgement as a property of the report
# ---------------------------------------------------------------------------

def test_a_clear_report_is_calm_not_an_alarm() -> None:
    findings = OutstandingFindings()
    findings.update("Findings: No acute cardiopulmonary process.")
    assert findings.count == 0
    assert findings.outstanding == ()
    assert findings.state == "clear"


def test_empty_text_has_nothing_outstanding() -> None:
    findings = OutstandingFindings()
    findings.update("")
    assert findings.count == 0 and findings.state == "clear"


def test_a_finding_starts_outstanding() -> None:
    findings = OutstandingFindings()
    findings.update(_URGENT_TEXT)
    assert findings.count == 1
    assert [f.term.lower() for f in findings.outstanding] == ["pneumothorax"]
    assert findings.state == "outstanding"


def test_acknowledging_settles_the_findings_in_the_report() -> None:
    findings = OutstandingFindings()
    findings.update(_URGENT_TEXT)
    findings.acknowledge()
    assert findings.outstanding == ()
    assert findings.count == 1
    assert findings.state == "acknowledged"


def test_an_unchanged_report_stays_acknowledged_after_a_rescan() -> None:
    # The scan runs on every edit; re-finding the same finding must not reopen a
    # question the radiologist has already answered.
    findings = OutstandingFindings()
    findings.update(_URGENT_TEXT)
    findings.acknowledge()
    findings.update(_URGENT_TEXT + " Lungs otherwise clear.")
    assert findings.state == "acknowledged"


def test_a_new_finding_typed_after_acknowledging_is_outstanding_again() -> None:
    # The defect this closes: acknowledgement used to be a one-shot event, so an
    # impression written after the dialog left the report unwarned at export.
    findings = OutstandingFindings()
    findings.update(_URGENT_TEXT)
    findings.acknowledge()
    findings.update(_URGENT_TEXT + "\nImpression: Acute appendicitis.")

    assert findings.count == 2
    assert [f.term.lower() for f in findings.outstanding] == ["acute appendicitis"]
    assert findings.state == "outstanding"


def test_deleting_one_finding_does_not_reopen_the_others() -> None:
    findings = OutstandingFindings()
    findings.update(_URGENT_TEXT + "\nImpression: Acute appendicitis.")
    findings.acknowledge()
    findings.update(_URGENT_TEXT)
    assert findings.state == "acknowledged" and findings.count == 1


def test_a_scanner_fault_leaves_nothing_outstanding(monkeypatch) -> None:
    # Fails open exactly as check_release does: a broken scanner must not put a
    # count on screen it cannot justify, nor block the report.
    def _boom(_text):
        raise RuntimeError("scanner exploded")

    monkeypatch.setattr(report_release, "scan_for_critical_findings", _boom)
    findings = OutstandingFindings()
    findings.update(_URGENT_TEXT)
    assert findings.count == 0 and findings.state == "clear"


def test_a_finding_carries_where_it_sits_in_the_text() -> None:
    # The gutter marks point at the finding by offset; searching for the words
    # again could land on a different occurrence.
    check = check_release(_URGENT_TEXT)
    found = check.findings[0]
    assert _URGENT_TEXT[found.start:found.end].lower() == "pneumothorax"


# ---------------------------------------------------------------------------
# record_release — the audit trail
# ---------------------------------------------------------------------------

def _finding(term: str, level: int) -> CriticalFinding:
    return CriticalFinding(term=term, level=level, negated=False, uncertain=False, context="…")


def _capture(monkeypatch) -> tuple[list, list]:
    acknowledged: list = []
    overrides: list = []
    monkeypatch.setattr(
        report_release.audit_log, "log_critical_finding_acknowledged",
        lambda term, patient_id, level: acknowledged.append((term, patient_id, level)),
    )
    monkeypatch.setattr(
        report_release.audit_log, "log_critical_finding_overridden",
        lambda terms, patient_id: overrides.append((terms, patient_id)),
    )
    return acknowledged, overrides


def test_acknowledgement_is_audited_once_per_finding(monkeypatch) -> None:
    acknowledged, overrides = _capture(monkeypatch)
    check = ReleaseCheck(
        findings=(_finding("pneumothorax", 1), _finding("appendicitis", 2)),
        summary="…", worst_level=1,
    )
    record_release(check, "P1", acknowledged=True)

    assert acknowledged == [("pneumothorax", "P1", 1), ("appendicitis", "P1", 2)]
    assert overrides == []


def test_proceeding_anyway_is_audited_once_as_an_override(monkeypatch) -> None:
    acknowledged, overrides = _capture(monkeypatch)
    check = ReleaseCheck(
        findings=(_finding("pneumothorax", 1), _finding("appendicitis", 2)),
        summary="…", worst_level=1,
    )
    record_release(check, "P1", acknowledged=False)

    assert overrides == [("pneumothorax; appendicitis", "P1")]
    assert acknowledged == []


def test_a_clear_report_writes_no_audit_entry(monkeypatch) -> None:
    acknowledged, overrides = _capture(monkeypatch)
    record_release(check_release("Findings: normal study."), "P1", acknowledged=False)
    assert acknowledged == [] and overrides == []


# ---------------------------------------------------------------------------
# Desktop coverage — every exit consults the gate
#
# This reads main_window.py's source rather than calling it: tests/conftest.py
# installs a PySide6 stub with only QtCore.QObject/Signal, so importing
# main_window (QtWidgets, QtGui) raises ModuleNotFoundError, and a real-Qt test
# would do nothing but skip. The defect being pinned is structural anyway — an
# exit that forgets the gate — so it is asserted structurally, and this runs
# every time instead of never.
# ---------------------------------------------------------------------------

_MAIN_WINDOW = Path(__file__).resolve().parents[1] / "src" / "ui" / "main_window.py"

#: Every way a report leaves the desktop app.
_GATED_EXITS = {"on_copy", "on_save_txt", "on_export_word"}


def _methods_calling(func_name: str) -> set[str]:
    tree = ast.parse(_MAIN_WINDOW.read_text(encoding="utf-8"))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(call.func, ast.Name) and call.func.id == func_name
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
        )
    }


def _methods_obeying(func_name: str) -> set[str]:
    """Methods containing ``if not <func_name>(...): return``."""
    tree = ast.parse(_MAIN_WINDOW.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for stmt in ast.walk(node):
            test = getattr(stmt, "test", None)
            if not isinstance(stmt, ast.If) or not isinstance(test, ast.UnaryOp):
                continue
            call = test.operand
            if (
                isinstance(test.op, ast.Not)
                and isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == func_name
                and any(isinstance(inner, ast.Return) for inner in stmt.body)
            ):
                found.add(node.name)
    return found


def test_every_desktop_exit_confirms_release() -> None:
    # The defect this replaces: the gate fired once when dictation stopped, so a
    # finding typed into the impression afterwards left the app unwarned.
    assert _GATED_EXITS <= _methods_calling("confirm_release")


def test_every_desktop_exit_stops_when_the_radiologist_cancels() -> None:
    # Calling the gate is not enough now that one of its two rules can say no:
    # an exit that ignores the answer would export the report anyway. Copy had
    # no unfilled-fields check at all before this.
    assert _GATED_EXITS <= _methods_obeying("confirm_release")


def test_autosave_is_not_gated() -> None:
    # Autosave stays inside the app's own directory; prompting on a timer is what
    # teaches people to dismiss the real warning.
    assert "_do_autosave" not in _methods_calling("confirm_release")
