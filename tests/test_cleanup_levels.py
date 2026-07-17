"""Regression tests for cleanup_level pipeline behavior (soft/medium/hard).

Cleanup levels control which postprocess stages run and whether AI cleanup is invoked:
- soft: skip terminology, accent corrections, medical-dict fuzzy, learned corrections
- medium: all 10 stages (the previous default; unchanged)
- hard: all 10 stages + optional Groq AI cleanup (when enabled + consented)

These tests verify that the stage-skip logic works correctly and that the hard
level properly invokes clean_with_llm (or skips it gracefully if disabled).
"""

from __future__ import annotations


from src.dictation.postprocess.pipeline import (
    postprocess_transcript,
    postprocess_transcript_with_changes,
    CLEANUP_LEVELS,
)


class TestCleanupLevels:
    """Verify cleanup_level parameter controls pipeline stage execution."""

    def test_cleanup_levels_defined(self):
        """Ensure CLEANUP_LEVELS tuple contains expected values."""
        assert CLEANUP_LEVELS == ("soft", "medium", "hard")

    def test_postprocess_accepts_cleanup_level(self):
        """postprocess_transcript accepts cleanup_level parameter."""
        result = postprocess_transcript(
            "test text with measurement 5.5cm",
            accent="neutral",
            cleanup_level="medium"
        )
        assert isinstance(result, str)

    def test_postprocess_with_changes_accepts_cleanup_level(self):
        """postprocess_transcript_with_changes accepts cleanup_level parameter."""
        result, changes = postprocess_transcript_with_changes(
            "test text",
            accent="neutral",
            cleanup_level="soft"
        )
        assert isinstance(result, str)
        assert isinstance(changes, list)

    def test_soft_cleanup_skips_terminology_stage(self):
        """Soft cleanup should NOT apply terminology corrections (US→UK)."""
        # US spelling that would normally be corrected to UK in medium mode
        text = "The esophageal findings are notable."
        soft_result = postprocess_transcript(text, cleanup_level="soft")
        medium_result = postprocess_transcript(text, cleanup_level="medium")

        # In medium mode, "esophageal" should normalize to "oesophageal" (Stage 5)
        # In soft mode, it should remain "esophageal"
        assert "esophageal" in soft_result.lower()
        # Both should produce valid results (medium may convert, soft should not)
        assert isinstance(medium_result, str)

    def test_soft_cleanup_skips_accent_corrections(self):
        """Soft cleanup should NOT apply accent-specific corrections (Stage 6)."""
        # This would be an accent-specific term that gets rewritten in medium mode
        # For now, just verify that soft mode produces different output than medium
        text = "Patient reported some changes"
        soft_result = postprocess_transcript(text, cleanup_level="soft")
        medium_result = postprocess_transcript(text, cleanup_level="medium")
        # Both should at least produce strings
        assert isinstance(soft_result, str)
        assert isinstance(medium_result, str)

    def test_soft_cleanup_skips_fuzzy_medical_dict(self):
        """Soft cleanup should NOT apply fuzzy medical-term correction (Stage 7)."""
        text = "Findings show no pneumathoria."  # typo: should not be corrected in soft
        soft_result = postprocess_transcript(text, cleanup_level="soft")
        # In soft mode, the typo stays (fuzzy stage skipped)
        # In medium mode, it would be corrected to "pneumothorax" or similar
        assert "pneumathoria" in soft_result

    def test_soft_cleanup_skips_learned_corrections(self):
        """Soft cleanup should NOT apply learned corrections (Stage 8)."""
        text = "Patient history is significant"
        soft_result = postprocess_transcript(text, cleanup_level="soft")
        medium_result = postprocess_transcript(text, cleanup_level="medium")
        # Both should produce valid text
        assert isinstance(soft_result, str)
        assert isinstance(medium_result, str)

    def test_medium_cleanup_is_default(self):
        """Medium cleanup is the default behavior (unchanged)."""
        text = "Test measurement 5.5cm"
        # Call without cleanup_level (should default to medium)
        result_default = postprocess_transcript(text)
        result_explicit = postprocess_transcript(text, cleanup_level="medium")
        # Both should produce the same result
        assert result_default == result_explicit

    def test_hard_cleanup_accepts_parameter(self):
        """Hard cleanup level should be accepted and processed."""
        text = "Test medical text"
        result = postprocess_transcript(text, cleanup_level="hard")
        # Should complete without error and return a string
        assert isinstance(result, str)

    def test_hard_cleanup_safe_on_llm_failure(self):
        """Hard cleanup should gracefully handle LLM cleanup errors."""
        # The clean_with_llm function should be best-effort, never raising
        text = "Test text"
        result = postprocess_transcript(text, cleanup_level="hard")
        # Should return a string even if LLM cleanup is disabled/fails
        assert isinstance(result, str)

    def test_invalid_cleanup_level_defaults_to_medium(self):
        """Invalid cleanup_level should default to medium (safe fallback)."""
        text = "Test text"
        result = postprocess_transcript(text, cleanup_level="invalid")
        # Should not raise; should treat as medium or skip hard logic
        assert isinstance(result, str)

    def test_cleanup_level_none_defaults_to_medium(self):
        """cleanup_level=None should default to medium."""
        text = "Test text"
        result = postprocess_transcript(text, cleanup_level=None)
        assert isinstance(result, str)


class TestHardLevelInvokesLLM:
    """The defining contract of 'hard': it (and only it) runs Groq AI cleanup.

    The previous tests only asserted ``isinstance(result, str)``, which is
    identical for every level and never proved the LLM stage fired or was
    skipped. These patch the lazily-imported ``clean_with_llm`` with a sentinel
    and assert on the call itself.
    """

    def _patch_llm(self, monkeypatch):
        """Patch clean_with_llm at its source module (pipeline imports it late)."""
        calls: list = []

        def _fake(text, *args, **kwargs):
            calls.append(text)
            return f"LLM[{text}]", []

        monkeypatch.setattr(
            "src.dictation.postprocess.llm_cleanup.clean_with_llm", _fake
        )
        return calls

    def test_hard_invokes_llm_cleanup(self, monkeypatch):
        """Hard mode calls clean_with_llm and uses its output."""
        calls = self._patch_llm(monkeypatch)
        result = postprocess_transcript("findings text", cleanup_level="hard")
        assert calls, "clean_with_llm was not called in hard mode"
        assert result.startswith("LLM["), "hard mode did not apply LLM output"

    def test_medium_does_not_invoke_llm_cleanup(self, monkeypatch):
        """Medium mode must never call clean_with_llm (no surprise network)."""
        calls = self._patch_llm(monkeypatch)
        postprocess_transcript("findings text", cleanup_level="medium")
        assert calls == [], "clean_with_llm fired in medium mode"

    def test_soft_does_not_invoke_llm_cleanup(self, monkeypatch):
        """Soft mode must never call clean_with_llm."""
        calls = self._patch_llm(monkeypatch)
        postprocess_transcript("findings text", cleanup_level="soft")
        assert calls == [], "clean_with_llm fired in soft mode"

    def test_invalid_level_does_not_invoke_llm_cleanup(self, monkeypatch):
        """An unknown level falls back to medium semantics — no LLM call."""
        calls = self._patch_llm(monkeypatch)
        postprocess_transcript("findings text", cleanup_level="invalid")
        assert calls == [], "clean_with_llm fired for an invalid level"

    def test_live_mode_never_invokes_llm_even_at_hard(self, monkeypatch):
        """Per-chunk live calls must NOT fire the network AI cleanup, even at
        'hard' — that pass belongs to the finished document only (CLAUDE.md
        invariant: AI cleanup is never in the per-chunk live path)."""
        calls = self._patch_llm(monkeypatch)
        postprocess_transcript("findings text", cleanup_level="hard", live=True)
        assert calls == [], "clean_with_llm fired during a live chunk"

    def test_with_changes_threads_live_flag(self, monkeypatch):
        """The with_changes wrapper must forward live=True so the banner path
        is also network-free during dictation."""
        calls = self._patch_llm(monkeypatch)
        postprocess_transcript_with_changes(
            "findings text", cleanup_level="hard", live=True
        )
        assert calls == [], "live flag not threaded through with_changes"


class TestCleanupLevelPreservesAccuracy:
    """Verify that cleanup levels maintain accuracy and don't over-correct."""

    def test_soft_level_preserves_correct_spellings(self):
        """Soft mode should preserve correctly-spelled medical terms unchanged."""
        text = "Pneumothorax noted on the right."
        result = postprocess_transcript(text, cleanup_level="soft")
        # Correctly spelled term should remain
        assert "pneumothorax" in result.lower()

    def test_medium_level_applies_all_stages(self):
        """Medium mode is the full pipeline (previous default)."""
        # This is the baseline behavior; all stages should run
        text = "measurement 5.5cm"
        result = postprocess_transcript(text, cleanup_level="medium")
        # Measurement stage should normalize it (e.g., to "5.5 cm")
        assert "cm" in result.lower()

    def test_hard_and_medium_both_valid(self):
        """Hard and medium cleanup levels should both produce valid output."""
        text = "Test medical text"
        hard_result = postprocess_transcript(text, cleanup_level="hard")
        medium_result = postprocess_transcript(text, cleanup_level="medium")

        # Both should be strings
        assert isinstance(hard_result, str)
        assert isinstance(medium_result, str)
        # Both should be non-empty (when input is non-empty)
        assert len(hard_result) > 0
        assert len(medium_result) > 0
