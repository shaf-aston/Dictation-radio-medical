"""Purposeful regression tests for the worker and post-processing pipeline.

Sliding-window / boundary-dedup coverage (window_start, advance_commit,
maybe_bootstrap, build_output) moved to tests/test_stream.py's segmenter and
ChunkLedger tests — window_state.py and text_diff.py, the modules that owned
that logic, were superseded by the chunk-once ledger (M2) and deleted.
"""

from __future__ import annotations

from runtime_stubs import install_test_runtime_stubs

install_test_runtime_stubs()

import src.features.adaptive_learning as adaptive_learning  # noqa: E402
import src.dictation.worker as transcribe_worker  # noqa: E402
from src.dictation.postprocess import (  # noqa: E402
    apply_correction_commands,
    filter_hallucinations,
    postprocess_transcript,
    postprocess_transcript_with_changes,
)
from src.dictation.worker import LiveTranscribeWorker, _RADIOLOGY_INITIAL_PROMPT  # noqa: E402


def make_worker() -> LiveTranscribeWorker:
    """Create a minimal worker instance without Qt thread setup."""
    worker = object.__new__(LiveTranscribeWorker)
    worker.pause_threshold = 2.5
    return worker


class TestContextPrompt:
    """Whisper context prompts should preserve useful context without bloat."""

    def test_uses_base_prompt_when_nothing_is_committed(self, monkeypatch) -> None:
        worker = make_worker()
        monkeypatch.setattr(transcribe_worker, "get_custom_prompt_suffix", lambda: "")

        assert worker._build_context_prompt() == _RADIOLOGY_INITIAL_PROMPT

    def test_appends_custom_terms_but_not_committed_text(self, monkeypatch) -> None:
        # Committed text intentionally must NOT appear in the prompt — when it
        # did, the live decoder echoed prior words back into the new window
        # under the small live beam. The prompt builder never reads committed
        # text (it lives in WindowState now), so it can't leak in.
        worker = make_worker()
        monkeypatch.setattr(
            transcribe_worker,
            "get_custom_prompt_suffix",
            lambda: "custom terms",
        )

        prompt = worker._build_context_prompt()

        assert prompt.startswith(_RADIOLOGY_INITIAL_PROMPT)
        assert prompt.endswith("custom terms")
        assert "word0" not in prompt
        assert "word199" not in prompt


class TestVoiceCorrectionCommands:
    """Voice editing commands should change only the intended words."""

    def test_inline_correction_replaces_the_preceding_word(self) -> None:
        assert apply_correction_commands("teh correct word the") == "the"

    def test_standalone_correction_targets_the_last_word_before_command(self) -> None:
        result = apply_correction_commands("there is teh correct word the finding")

        assert result == "there is the finding"


class TestHallucinationFiltering:
    """Known Whisper junk should be removed cleanly."""

    def test_filters_common_thank_you_hallucinations(self) -> None:
        assert filter_hallucinations("thank you for watching") == ""

    def test_filters_repetition_only_lines(self) -> None:
        assert filter_hallucinations("word word word word word") == ""


class TestPostProcessingPipeline:
    """High-value end-to-end regressions across the ordered pipeline."""

    def test_context_aware_punctuation_preserves_anatomical_colon(self) -> None:
        assert postprocess_transcript("ascending colon is normal") == "Ascending colon is normal"

    def test_spoken_colon_becomes_punctuation(self) -> None:
        assert postprocess_transcript("findings colon impression") == "Findings: impression"

    def test_context_aware_period_preserves_medical_usage(self) -> None:
        assert postprocess_transcript("menstrual period is normal") == "Menstrual period is normal"

    def test_spoken_period_becomes_punctuation(self) -> None:
        assert postprocess_transcript("end period") == "End."

    def test_measurements_are_standardised(self) -> None:
        assert postprocess_transcript("5 by 3 by 2 millimetres") == "5 x 3 x 2 mm"
        assert postprocess_transcript("15 degrees") == "15°"

    def test_msk_corrections_stay_context_aware(self) -> None:
        assert postprocess_transcript("anterior crucial ligament") == "Anterior cruciate ligament"
        assert postprocess_transcript("this finding is crucial") == "This finding is crucial"
        assert postprocess_transcript("median meniscus") == "Medial meniscus"
        assert postprocess_transcript("the median value is five") == "The median value is five"

    def test_common_whisper_msk_error_is_corrected(self) -> None:
        assert postprocess_transcript("there is a tier of the supraspinatus") == (
            "There is a tear of the supraspinatus"
        )

    def test_accent_specific_corrections_only_run_for_selected_profile(self) -> None:
        assert postprocess_transcript("wertebra", accent="south_asian") == "Vertebra"
        assert postprocess_transcript("wertebra", accent="neutral") == "Wertebra"

    def test_learned_corrections_override_pipeline_output_last(self) -> None:
        adaptive_learning.get_adaptive_learning().learn_correction("tear", "rupture")

        assert postprocess_transcript("meniscal tier") == "Meniscal rupture"

    def test_change_summary_reports_meaningful_replacements(self) -> None:
        processed, changes = postprocess_transcript_with_changes("meniscal tier")

        assert processed == "Meniscal tear"
        assert changes == ['"meniscal tier" → "Meniscal tear"']

    def test_change_diff_is_skipped_on_very_long_documents(self) -> None:
        # The word-level diff is quadratic (~2 s at ~2,200 words — measured),
        # so above _MAX_DIFF_WORDS the banner examples are dropped. The text
        # itself must still be fully corrected; only the diff is skipped.
        doc = "meniscal tier " * 1600  # 3,200 words, over the bound

        processed, changes = postprocess_transcript_with_changes(doc)

        assert changes == []
        assert "tear" in processed
        assert "tier" not in processed
