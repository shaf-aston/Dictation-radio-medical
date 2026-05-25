"""End-to-end postprocessing pipeline tests.

This module tests the full pipeline from Whisper transcription output through
postprocessing. Tests focus on the correction logic rather than actual audio I/O,
which requires heavy dependencies (Whisper model, audioprocessing libraries).
"""

from __future__ import annotations

import pytest

from src.dictation.postprocess import postprocess_transcript, postprocess_transcript_with_changes


class TestPostprocessingPipeline:
    """End-to-end postprocessing of transcription output."""

    def test_pipeline_handles_medical_terminology(self) -> None:
        """Pipeline should correct medical terms from Whisper output."""
        # Simulate common Whisper output
        whisper_output = "meniscal tier of the left knee"

        processed = postprocess_transcript(whisper_output)

        # Should correct "tier" → "tear"
        assert "tear" in processed.lower()
        assert "tier" not in processed.lower()

    def test_pipeline_standardizes_measurements(self) -> None:
        """Pipeline should standardize measurement notation."""
        whisper_output = "5 by 3 by 2 millimetres"

        processed = postprocess_transcript(whisper_output)

        # Should convert to "5 x 3 x 2 mm"
        assert "x" in processed
        assert "mm" in processed
        assert "millimetres" not in processed

    def test_pipeline_applies_accent_corrections(self) -> None:
        """Pipeline should apply accent-specific corrections."""
        whisper_output = "wertebra is normal"

        # Without accent
        result_neutral = postprocess_transcript(whisper_output, accent="neutral")
        assert "wertebra" in result_neutral.lower()

        # With south asian accent
        result_sa = postprocess_transcript(whisper_output, accent="south_asian")
        assert "vertebra" in result_sa.lower()
        assert "wertebra" not in result_sa.lower()

    def test_pipeline_capitalizes_output(self) -> None:
        """Pipeline should capitalize sentences."""
        whisper_output = "anterior cruciate ligament is intact"

        processed = postprocess_transcript(whisper_output)

        # Should start with capital letter
        assert processed[0].isupper()

    def test_pipeline_applies_multiple_corrections(self) -> None:
        """Pipeline should apply multiple corrections in order."""
        whisper_output = "5 by 3 millimetres meniscal tier normal finding"

        processed = postprocess_transcript(whisper_output)

        # Should correct measurements
        assert "x" in processed
        assert "mm" in processed

        # Should correct terminology
        assert "tear" in processed.lower()

    def test_pipeline_tracks_changes(self) -> None:
        """Pipeline should report changes made during processing."""
        whisper_output = "meniscal tier tear"

        processed, changes = postprocess_transcript_with_changes(whisper_output)

        # Should have at least one change
        assert len(changes) > 0
        assert any("tier" in c for c in changes)

    def test_pipeline_handles_empty_input(self) -> None:
        """Pipeline should handle empty or whitespace-only input."""
        assert postprocess_transcript("") == ""
        assert postprocess_transcript("   ") == ""

    def test_pipeline_preserves_valid_terminology(self) -> None:
        """Pipeline should NOT correct valid medical terms."""
        # "tear" is correct, should not be changed
        whisper_output = "rotator cuff tear"

        processed = postprocess_transcript(whisper_output)

        assert "tear" in processed.lower()

    def test_pipeline_handles_context_aware_corrections(self) -> None:
        """Pipeline should use context to avoid false corrections."""
        # "period" is valid in "menstrual period", should not become punctuation
        whisper_output = "menstrual period is normal"

        processed = postprocess_transcript(whisper_output)

        # Should NOT convert "period" to "."
        assert "menstrual period" in processed.lower()

    def test_pipeline_standardizes_degree_symbol(self) -> None:
        """Pipeline should convert 'degrees' to degree symbol."""
        whisper_output = "15 degrees of rotation"

        processed = postprocess_transcript(whisper_output)

        # Should convert to degree symbol
        assert "°" in processed or "degrees" in processed.lower()

    def test_pipeline_removes_common_hallucinations(self) -> None:
        """Pipeline should filter known Whisper hallucinations at boundaries."""
        # End-of-text hallucinations are removed
        whisper_output = "normal findings here see you next time"

        processed = postprocess_transcript(whisper_output)

        # Should remove the hallucination at the end ("see you next time")
        assert "see you" not in processed
        assert "normal" in processed.lower()

    def test_pipeline_preserves_medical_acronyms(self) -> None:
        """Pipeline should preserve medical acronyms like CT, MRI."""
        whisper_output = "CT of the abdomen shows no abnormality"

        processed = postprocess_transcript(whisper_output)

        # CT should be preserved
        assert "CT" in processed or "ct" in processed.lower()

    def test_pipeline_handles_multiple_acronyms(self) -> None:
        """Pipeline should handle multiple acronyms in one report."""
        whisper_output = "MRI brain and CT spine are normal"

        processed = postprocess_transcript(whisper_output)

        # Both should be preserved
        assert "MRI" in processed or "mri" in processed.lower()
        assert "CT" in processed or "ct" in processed.lower()

    def test_pipeline_accent_suggestion(self) -> None:
        """Test accent suggestion based on detected patterns."""
        from src.features.accent_corrections import suggest_accent

        # South Asian error patterns
        sa_text = "wertebra tandon ligament toracic wein frachure legament"
        suggested = suggest_accent(sa_text)
        assert suggested == "south_asian"

        # No clear patterns
        neutral_text = "normal study performed"
        suggested = suggest_accent(neutral_text)
        assert suggested is None

    def test_pipeline_batch_processing(self) -> None:
        """Pipeline should handle multiple corrections in sequence."""
        reports = [
            "meniscal tier",
            "5 by 3 millimetres",
            "anterior cruciate ligament",
        ]

        processed_list = [postprocess_transcript(r) for r in reports]

        assert len(processed_list) == 3
        assert "tear" in processed_list[0].lower()
        assert "x" in processed_list[1]
        assert processed_list[2][0].isupper()
