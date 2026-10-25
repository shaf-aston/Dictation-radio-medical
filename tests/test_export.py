"""What leaves as a file is the report, whatever characters it holds."""

import io

import pytest

from src.features.report_manager import export_to_word_bytes, format_plain_text_report

docx = pytest.importorskip("docx")

REPORT = ("FINDINGS:\nFindings are unremarkable apart from a small effusion.\n"
          "Comparison with the prior study shows no change.\n    - indented sub-point\n"
          "IMPRESSION: Small effusion.")


def _paragraphs(data: bytes):
    return [(p.style.name, p.text) for p in docx.Document(io.BytesIO(data)).paragraphs if p.text]


def test_only_a_whole_section_name_is_a_heading():
    paragraphs = dict((text, style) for style, text in _paragraphs(export_to_word_bytes(REPORT, {})))
    assert paragraphs["FINDINGS:"].startswith("Heading")
    assert paragraphs["Findings are unremarkable apart from a small effusion."] == "Normal"
    assert paragraphs["Comparison with the prior study shows no change."] == "Normal"
    assert paragraphs["IMPRESSION: Small effusion."] == "Normal"     # label run is bold
    assert "    - indented sub-point" in paragraphs


@pytest.mark.parametrize("bad", ["\x0b", "\x0c", "\x1b"])
def test_a_control_character_never_breaks_either_export(bad):
    patient = {"id": f"MRN{bad}123", "name": "Jane"}
    text = f"No effusion.{bad} Normal study."
    assert "No effusion." in " ".join(t for _, t in _paragraphs(export_to_word_bytes(text, patient)))
    assert bad not in format_plain_text_report(text, patient)
