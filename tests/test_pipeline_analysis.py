"""Regression tests for the post-processing PipelineAnalyzer writer.

Guards the bug where ``save_analysis`` wrote without ``encoding="utf-8"``: on a
non-UTF-8 locale (e.g. Windows cp1252) any accented/medical-unicode transcript
raised ``UnicodeEncodeError`` mid-write, leaving a corrupt analysis file.
"""

from __future__ import annotations

import json

from src.dictation.postprocess import analysis


def test_save_analysis_handles_non_ascii(tmp_path, monkeypatch):
    """Transcript text outside cp1252 must round-trip without raising."""
    monkeypatch.setattr(analysis, "analysis_dir", lambda: tmp_path)

    pa = analysis.PipelineAnalyzer()
    # µ, °, –, and Cyrillic are all outside cp1252 / would break a locale write.
    pa.record_input("Lesion 5µm at 37° — измерение")
    pa.record_stage("terminology", "Lesion 5 µm at 37 ° — измерение", 0.01)
    pa.save_analysis(accent="neutral")

    files = list(tmp_path.glob("analysis_*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert data["input"] == "Lesion 5µm at 37° — измерение"
    assert data["accent"] == "neutral"
    assert data["stages"][0]["stage"] == "terminology"
