"""NegEx-style critical-findings detection regression tests."""

from __future__ import annotations

from src.medical.critical_findings import (
    CriticalFinding,
    format_findings_for_dialog,
    scan_for_critical_findings,
)


class TestPositiveDetection:
    """Unambiguous positive findings should be reported with the correct level."""

    def test_pneumothorax_is_flagged_as_level_one(self) -> None:
        findings = scan_for_critical_findings("There is a large pneumothorax on the right.")
        assert any(f.level == 1 and "pneumothorax" in f.term.lower() for f in findings)

    def test_appendicitis_is_flagged_as_level_two(self) -> None:
        findings = scan_for_critical_findings("Findings consistent with acute appendicitis.")
        assert any(f.level == 2 for f in findings)

    def test_findings_are_sorted_level_one_first(self) -> None:
        findings = scan_for_critical_findings(
            "Acute appendicitis. Also evidence of subdural haematoma."
        )
        assert findings, "expected at least one finding"
        assert findings[0].level == 1


class TestNegationParsing:
    """Cleanly negated findings should be suppressed; uncertainty should be kept."""

    def test_clean_negation_suppresses_level_two_finding(self) -> None:
        findings = scan_for_critical_findings("No evidence of pulmonary embolism.")
        assert all("pulmonary embolism" not in f.term.lower() for f in findings)

    def test_uncertainty_phrase_still_flags_finding(self) -> None:
        findings = scan_for_critical_findings("Cannot exclude pulmonary embolism.")
        assert any(f.uncertain and "pulmonary embolism" in f.term.lower() for f in findings)

    def test_post_finding_negation_suppresses_finding(self) -> None:
        findings = scan_for_critical_findings("Pulmonary embolism is not seen.")
        assert all("pulmonary embolism" not in f.term.lower() for f in findings)

    def test_concerning_for_phrase_marks_finding_uncertain(self) -> None:
        findings = scan_for_critical_findings("Concerning for acute appendicitis.")
        uncertain = [f for f in findings if "appendicitis" in f.term.lower() and f.uncertain]
        assert uncertain, "expected at least one uncertain appendicitis finding"


class TestDialogFormatting:
    """The dialog formatter should render severity, term, and context."""

    def test_format_includes_severity_and_term(self) -> None:
        finding = CriticalFinding(
            term="pneumothorax",
            level=1,
            negated=False,
            uncertain=False,
            context="...left pneumothorax...",
        )
        out = format_findings_for_dialog([finding])
        assert "LIFE-THREATENING" in out
        assert "PNEUMOTHORAX" in out
        assert "Context:" in out

    def test_uncertain_flag_shows_qualifier(self) -> None:
        finding = CriticalFinding(
            term="pulmonary embolism",
            level=2,
            negated=False,
            uncertain=True,
            context="cannot exclude pulmonary embolism",
        )
        out = format_findings_for_dialog([finding])
        assert "UNCERTAIN" in out
        assert "URGENT" in out

    def test_empty_findings_returns_empty_string(self) -> None:
        assert format_findings_for_dialog([]) == ""
