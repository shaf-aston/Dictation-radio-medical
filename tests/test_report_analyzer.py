"""Tests for local report analysis (requirement: terms / templates / regions).

The critical regression these guard against: the report body must be extracted
from the formatted report, with the patient header and the footer (report date,
radiologist line) excluded — otherwise the statistics mine boilerplate/PHI
instead of the clinical findings.
"""

from __future__ import annotations

import pytest

from src.features.file_manager import autosave_dir
from src.features.report_manager import format_plain_text_report
from src.features.report_analyzer import ReportAnalyzer

_PHI = {"name": "John Smith", "id": "AB1234567"}


def _write_report(body: str, idx: int) -> None:
    path = autosave_dir() / f"rep{idx}.txt"
    path.write_text(format_plain_text_report(body, _PHI), encoding="utf-8")


def test_strip_header_returns_body_only():
    raw = format_plain_text_report(
        "FINDINGS:\nThe ACL is intact.\n\nIMPRESSION:\nNormal.", _PHI
    )
    body = ReportAnalyzer._strip_header(raw)
    assert "ACL is intact" in body
    # Patient header and footer boilerplate must be gone.
    assert "John Smith" not in body
    assert "AB1234567" not in body
    assert "Reported:" not in body
    assert "Radiologist" not in body


def test_top_terms_mine_findings_not_boilerplate():
    bodies = [
        "FINDINGS:\nThe ACL is intact. The medial meniscus shows a tear.\n\nIMPRESSION:\nMeniscal tear.",
        "FINDINGS:\nThe ACL is intact. The lateral meniscus shows a tear.\n\nIMPRESSION:\nMeniscal tear.",
        "FINDINGS:\nThe ACL is intact. Small joint effusion.\n\nIMPRESSION:\nEffusion.",
    ]
    for i, b in enumerate(bodies):
        _write_report(b, i)

    a = ReportAnalyzer().analyze_all_reports()
    assert a.report_count == 3
    terms = {t for t, _ in a.top_terms}
    assert "acl" in terms
    assert any("menisc" in t for t in terms)
    # No PHI or footer terms leaked into the statistics.
    assert "smith" not in terms and "radiologist" not in terms and "reported" not in terms


def test_recurring_sentence_becomes_template():
    for i in range(3):
        _write_report("FINDINGS:\nThe ACL is intact.\n\nIMPRESSION:\nNormal study.", i)
    a = ReportAnalyzer().analyze_all_reports()
    suggested = " ".join(s for s, _ in a.template_suggestions)
    assert "ACL is intact" in suggested


def test_body_region_distribution():
    _write_report("FINDINGS:\nThe medial meniscus and ACL are normal.", 0)
    _write_report("FINDINGS:\nThe rotator cuff and supraspinatus are intact.", 1)
    a = ReportAnalyzer().analyze_all_reports()
    assert a.body_region_distribution.get("knee") == 1
    assert a.body_region_distribution.get("shoulder") == 1


def test_no_reports_yields_empty_analytics():
    a = ReportAnalyzer().analyze_all_reports()
    assert a.report_count == 0
    assert a.top_terms == []
