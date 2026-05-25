"""Purposeful regression tests for the worker and post-processing pipeline."""

from __future__ import annotations

import numpy as np

from runtime_stubs import install_test_runtime_stubs

install_test_runtime_stubs()

import src.features.adaptive_learning as adaptive_learning  # noqa: E402
import src.workers.transcribe_worker as transcribe_worker  # noqa: E402
from src.core.postprocess import (  # noqa: E402
    apply_correction_commands,
    filter_hallucinations,
    postprocess_transcript,
    postprocess_transcript_with_changes,
)
from src.workers.transcribe_worker import (  # noqa: E402
    LiveTranscribeWorker,
    _OVERLAP_SEC,
    _RADIOLOGY_INITIAL_PROMPT,
    _WINDOW_SEC,
)

SR = 16000


def make_worker() -> LiveTranscribeWorker:
    """Create a minimal worker instance without Qt thread setup."""
    worker = object.__new__(LiveTranscribeWorker)
    worker._committed_text = ""
    worker._committed_samples = 0
    worker._last_emitted = ""
    worker._prev_total_samples = 0
    worker._prev_segments = []
    worker._prev_chunk_start_sec = 0.0
    worker.pause_threshold = 2.5
    return worker


def silence(seconds: float) -> np.ndarray:
    """Generate silent audio at the sample rate Whisper expects."""
    return np.zeros(int(seconds * SR), dtype=np.float32)


def segment(start: float, end: float, text: str) -> dict:
    """Build a transcription segment payload."""
    return {"start": start, "end": end, "text": text}


class TestWindowing:
    """The sliding window should stay bounded and predictable."""

    def test_short_audio_is_returned_whole(self) -> None:
        worker = make_worker()
        audio = silence(_WINDOW_SEC - 1)

        chunk, start_sec = worker._window(audio, SR, len(audio) / SR)

        assert len(chunk) == len(audio)
        assert start_sec == 0.0

    def test_first_long_window_is_capped_to_recent_audio(self) -> None:
        worker = make_worker()
        audio = silence(_WINDOW_SEC + 10)

        chunk, start_sec = worker._window(audio, SR, len(audio) / SR)

        assert len(chunk) / SR == _WINDOW_SEC
        assert start_sec == 10.0

    def test_commit_anchor_still_respects_hard_cap(self) -> None:
        worker = make_worker()
        worker._committed_samples = int(10.0 * SR)
        audio = silence(40.0)

        chunk, start_sec = worker._window(audio, SR, 40.0)

        assert len(chunk) / SR == _WINDOW_SEC
        assert start_sec == 15.0

    def test_commit_near_tail_uses_overlap_context(self) -> None:
        worker = make_worker()
        worker._committed_samples = int(29.0 * SR)
        audio = silence(40.0)

        _, start_sec = worker._window(audio, SR, 40.0)

        assert start_sec == 29.0 - _OVERLAP_SEC


class TestBootstrapCommit:
    """Window slides should commit only text that has safely expired."""

    def test_commits_only_segments_before_new_window(self) -> None:
        worker = make_worker()
        segments = [
            segment(1.0, 4.0, "The ACL appears intact"),
            segment(4.5, 6.5, "This should stay live"),
        ]

        worker._bootstrap_commit(segments, 0.0, 5.8, SR)

        assert worker._committed_text == "The ACL appears intact"
        assert worker._committed_samples == int(4.0 * SR)

    def test_respects_previous_window_offset(self) -> None:
        worker = make_worker()

        worker._bootstrap_commit([segment(0.5, 1.5, "word")], 2.0, 3.8, SR)

        assert worker._committed_text == "word"
        assert worker._committed_samples == int(3.5 * SR)


class TestBuildOutput:
    """Final output should avoid duplicated boundary text."""

    def test_returns_chunk_text_before_any_commit(self) -> None:
        worker = make_worker()

        assert worker._build_output("chunk result", 0.0) == "chunk result"

    def test_deduplicates_boundary_overlap_using_text_match(self) -> None:
        worker = make_worker()
        worker._committed_text = "Committed part overlap text"
        worker._committed_samples = int(5.8 * SR)

        output = worker._build_output(
            "overlap text new findings",
            5.8 - _OVERLAP_SEC,
        )

        assert output == "Committed part overlap text new findings"

    def test_falls_back_to_simple_append_when_no_overlap_is_found(self) -> None:
        worker = make_worker()
        worker._committed_text = "Committed part."
        worker._committed_samples = int(4.0 * SR)

        output = worker._build_output("completely different text", 5.8)

        assert output == "Committed part. completely different text"


class TestAdvanceCommit:
    """Stable text should be committed once it falls behind the safe frontier."""

    def test_commits_only_segments_behind_safe_frontier(self) -> None:
        worker = make_worker()
        segments = [
            segment(0.2, 1.0, "alpha"),
            segment(1.2, 2.0, "beta"),
            segment(2.3, 4.5, "gamma"),
        ]

        worker._advance_commit(segments, 5.0, 32.0, SR)

        assert worker._committed_text == "alpha beta"
        assert worker._committed_samples == int(7.0 * SR)

    def test_does_nothing_when_frontier_has_not_advanced(self) -> None:
        worker = make_worker()
        worker._committed_text = "already committed"
        worker._committed_samples = int(8.0 * SR)

        worker._advance_commit([segment(0.0, 1.0, "new text")], 5.0, 32.0, SR)

        assert worker._committed_text == "already committed"
        assert worker._committed_samples == int(8.0 * SR)


class TestContextPrompt:
    """Whisper context prompts should preserve useful context without bloat."""

    def test_uses_base_prompt_when_nothing_is_committed(self, monkeypatch) -> None:
        worker = make_worker()
        monkeypatch.setattr(transcribe_worker, "get_custom_prompt_suffix", lambda: "")

        assert worker._build_context_prompt() == _RADIOLOGY_INITIAL_PROMPT

    def test_appends_custom_terms_but_not_committed_text(self, monkeypatch) -> None:
        # Committed text intentionally must NOT appear in the prompt — when it
        # did, the live decoder echoed prior words back into the new window
        # under the small live beam.
        worker = make_worker()
        worker._committed_text = " ".join(f"word{i}" for i in range(200))
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
