"""Purposeful regression tests for the worker and post-processing pipeline."""

from __future__ import annotations

import numpy as np

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
from src.dictation.worker import (  # noqa: E402
    LiveTranscribeWorker,
    _COMMIT_LAG_SEC,  # type: ignore
    _RADIOLOGY_INITIAL_PROMPT,
    _WINDOW_SEC,  # type: ignore
)
from src.dictation.window_state import WindowState, _OVERLAP_SEC  # noqa: E402

SR = 16000


def make_state() -> WindowState:
    """The pure sliding-window / commit machine — no Qt, no I/O."""
    return WindowState(_WINDOW_SEC, _COMMIT_LAG_SEC)


def make_worker() -> LiveTranscribeWorker:
    """Create a minimal worker instance without Qt thread setup."""
    worker = object.__new__(LiveTranscribeWorker)
    worker.pause_threshold = 2.5
    return worker


def silence(seconds: float) -> np.ndarray:
    """Generate silent audio at the sample rate Whisper expects."""
    return np.zeros(int(seconds * SR), dtype=np.float32)


def segment(start: float, end: float, text: str) -> dict:
    """Build a transcription segment payload."""
    return {"start": start, "end": end, "text": text}


def window(state: WindowState, total_sec: float) -> tuple[float, float]:
    """Return ``(chunk_seconds, start_seconds)`` for a recording of *total_sec*.

    Only this slice is read off disk rather than decoding the whole growing
    WAV, so the window is expressed as sample offsets.
    """
    total_samples = int(total_sec * SR)
    start = state.window_start(total_samples, SR)
    return (total_samples - start) / SR, start / SR


class TestWindowGeometry:
    """WindowState clamps its knobs to safe, lossless bounds."""

    def test_clamps_commit_lag_below_window_minus_overlap(self) -> None:
        # A commit lag larger than the window is nonsensical; it must be pulled
        # back so the window can always re-cover the committed tail.
        state = WindowState(window_sec=10.0, commit_lag_sec=100.0)
        assert state.commit_lag_sec == 10.0 - _OVERLAP_SEC

    def test_clamps_commit_lag_above_overlap(self) -> None:
        # A commit lag below the overlap would let text commit that the window
        # no longer re-covers — word loss. Floor it at the overlap.
        state = WindowState(window_sec=25.0, commit_lag_sec=0.0)
        assert state.commit_lag_sec == _OVERLAP_SEC


class TestWindowing:
    """The sliding window should stay bounded and predictable."""

    def test_short_audio_is_returned_whole(self) -> None:
        state = make_state()

        chunk_sec, start_sec = window(state, _WINDOW_SEC - 1)

        assert chunk_sec == _WINDOW_SEC - 1
        assert start_sec == 0.0

    def test_first_long_window_is_capped_to_recent_audio(self) -> None:
        state = make_state()

        chunk_sec, start_sec = window(state, _WINDOW_SEC + 10)

        assert chunk_sec == _WINDOW_SEC
        assert start_sec == 10.0

    def test_commit_anchor_still_respects_hard_cap(self) -> None:
        state = make_state()
        state.committed_samples = int(10.0 * SR)

        chunk_sec, start_sec = window(state, 40.0)

        assert chunk_sec == _WINDOW_SEC
        assert start_sec == 15.0

    def test_commit_near_tail_uses_overlap_context(self) -> None:
        state = make_state()
        state.committed_samples = int(29.0 * SR)

        _, start_sec = window(state, 40.0)

        assert start_sec == 29.0 - _OVERLAP_SEC

    def test_steady_state_window_is_small_not_full(self) -> None:
        # Once text commits, the window is just the un-committed tail plus
        # overlap (commit_lag + overlap), NOT a full _WINDOW_SEC — this is the
        # core live-speed fix. Frontier sits commit_lag behind "now".
        state = make_state()
        total = 60.0
        state.committed_samples = int((total - _COMMIT_LAG_SEC) * SR)

        chunk_sec, _ = window(state, total)

        assert chunk_sec == _COMMIT_LAG_SEC + _OVERLAP_SEC
        assert chunk_sec < _WINDOW_SEC

    def test_window_re_covers_committed_tail_so_no_gap(self) -> None:
        # The window must start at or before the commit frontier, so the
        # just-committed tail is re-transcribed and de-duplicated rather than
        # dropped. This is the losslessness invariant behind aggressive commit.
        state = make_state()
        total = 60.0
        frontier_sec = total - _COMMIT_LAG_SEC
        state.committed_samples = int(frontier_sec * SR)

        _, start_sec = window(state, total)

        assert start_sec <= frontier_sec
        assert frontier_sec - start_sec == _OVERLAP_SEC


class TestBootstrapCommit:
    """Window slides should commit only text that has safely expired."""

    def test_commits_only_segments_before_new_window(self) -> None:
        state = make_state()
        segments = [
            segment(1.0, 4.0, "The ACL appears intact"),
            segment(4.5, 6.5, "This should stay live"),
        ]

        state.record_segments(segments, 0.0)
        state.maybe_bootstrap(5.8, SR)

        assert state.committed_text == "The ACL appears intact"
        assert state.committed_samples == int(4.0 * SR)

    def test_respects_previous_window_offset(self) -> None:
        state = make_state()

        state.record_segments([segment(0.5, 1.5, "word")], 2.0)
        state.maybe_bootstrap(3.8, SR)

        assert state.committed_text == "word"
        assert state.committed_samples == int(3.5 * SR)

    def test_noop_when_window_has_not_moved(self) -> None:
        state = make_state()
        state.record_segments([segment(0.5, 1.5, "word")], 2.0)

        state.maybe_bootstrap(2.2, SR)  # < prev_start + 0.5 → no commit

        assert state.committed_text == ""
        assert state.committed_samples == 0


class TestBuildOutput:
    """Final output should avoid duplicated boundary text."""

    def test_returns_chunk_text_before_any_commit(self) -> None:
        state = make_state()

        assert state.build_output("chunk result", 0.0) == "chunk result"

    def test_deduplicates_boundary_overlap_using_text_match(self) -> None:
        state = make_state()
        state.committed_text = "Committed part overlap text"
        state.committed_samples = int(5.8 * SR)

        output = state.build_output(
            "overlap text new findings",
            5.8 - _OVERLAP_SEC,
        )

        assert output == "Committed part overlap text new findings"

    def test_falls_back_to_simple_append_when_no_overlap_is_found(self) -> None:
        state = make_state()
        state.committed_text = "Committed part."
        state.committed_samples = int(4.0 * SR)

        output = state.build_output("completely different text", 5.8)

        assert output == "Committed part. completely different text"

    def test_committed_prefix_len_locates_frozen_prefix(self) -> None:
        state = make_state()
        state.committed_text = "Frozen prefix"

        assert state.committed_prefix_len("Frozen prefix and more") == len(
            "Frozen prefix"
        )
        assert state.committed_prefix_len("different text") == 0


class TestAdvanceCommit:
    """Stable text should be committed once it falls behind the safe frontier."""

    def test_commits_only_segments_behind_safe_frontier(self) -> None:
        state = make_state()
        segments = [
            segment(0.2, 1.0, "alpha"),
            segment(1.2, 2.0, "beta"),
            segment(2.3, 4.5, "gamma"),
        ]

        # total=16 → safe frontier = 16 - commit_lag(8) = 8s; gamma ends at
        # abs 9.5s so it stays live, alpha+beta (end 7s) commit.
        state.advance_commit(segments, 5.0, 16.0, SR)

        assert state.committed_text == "alpha beta"
        assert state.committed_samples == int(7.0 * SR)

    def test_does_nothing_when_frontier_has_not_advanced(self) -> None:
        state = make_state()
        state.committed_text = "already committed"
        state.committed_samples = int(8.0 * SR)

        state.advance_commit([segment(0.0, 1.0, "new text")], 5.0, 32.0, SR)

        assert state.committed_text == "already committed"
        assert state.committed_samples == int(8.0 * SR)


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
