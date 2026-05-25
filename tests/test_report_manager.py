"""Report manager: plain-text formatting, save, and autosave."""

from __future__ import annotations

from pathlib import Path


from src.features import file_manager, report_manager


PATIENT = {
    "id": "12345",
    "name": "Jane Doe",
    "dob": "01/01/1980",
    "study_date": "04/05/2026",
    "referring": "Dr Smith",
    "accession": "ACC-001",
}
BODY = "FINDINGS:\nNo acute abnormality."


class TestPlainTextFormatting:
    """The plain-text header should include every patient field and the body."""

    def test_header_contains_each_patient_field(self) -> None:
        out = report_manager._format_plain_text(BODY, PATIENT)
        for value in PATIENT.values():
            assert value in out

    def test_body_appears_after_header(self) -> None:
        out = report_manager._format_plain_text(BODY, PATIENT)
        assert "FINDINGS:" in out
        assert "RADIOLOGY REPORT" in out
        assert out.index("RADIOLOGY REPORT") < out.index("FINDINGS:")


class TestSaveTxt:
    """save_report_txt should write the formatted report to the given path."""

    def test_writes_formatted_report(self, tmp_path: Path) -> None:
        out = tmp_path / "report.txt"
        report_manager.save_report_txt(str(out), BODY, PATIENT)

        contents = out.read_text(encoding="utf-8")
        assert "FINDINGS:" in contents
        assert PATIENT["name"] in contents


class TestAutosave:
    """autosave_report should drop a file in autosave_dir() with patient ID in name."""

    def test_returns_none_for_blank_text(self) -> None:
        assert report_manager.autosave_report("   \n", PATIENT) is None

    def test_writes_file_and_returns_path(self) -> None:
        path = report_manager.autosave_report(BODY, PATIENT)
        assert path is not None
        saved = Path(path)
        assert saved.exists()
        assert saved.parent == file_manager.autosave_dir()
        assert PATIENT["id"] in saved.name

    def test_unknown_patient_id_is_substituted(self) -> None:
        path = report_manager.autosave_report(BODY, {"id": ""})
        assert path is not None
        assert "unknown" in Path(path).name
