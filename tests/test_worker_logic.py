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
from src.dictation.worker import (  # noqa: E402
    RADIOLOGY_PROMPT,
    LiveTranscribeWorker,
    should_skip_preview,
)


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

        assert worker._build_context_prompt() == RADIOLOGY_PROMPT

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

        # Learned terms first, shipped vocabulary last. Whisper keeps only the
        # LAST 223 tokens of a prompt, so the text written last is the text that
        # survives — see tests/test_prompt_budget.py.
        assert prompt.startswith("custom terms")
        assert prompt.endswith(RADIOLOGY_PROMPT)
        assert "word0" not in prompt
        assert "word199" not in prompt


class TestPreviewSkip:
    """When the live preview is worth its decode time, and when it is not.

    The preview is never committed, so dropping it cannot change the report —
    it only decides whether the next second of CPU goes to words that are kept
    or to words that are about to be replaced anyway.
    """

    def test_a_machine_that_keeps_up_always_keeps_the_preview(self) -> None:
        # decode_cost below 1.0 = decodes faster than speech arrives, so there
        # is slack to spend and no reason to blank the live text.
        assert should_skip_preview(open_tail_sec=19.0, decode_cost=0.4, max_lag_sec=3.0) is False

    def test_a_slow_machine_drops_the_preview_once_the_tail_is_long(self) -> None:
        assert should_skip_preview(open_tail_sec=5.0, decode_cost=3.0, max_lag_sec=3.0) is True

    def test_a_slow_machine_keeps_a_short_tail_preview(self) -> None:
        # Just after a chunk closed: the tail is cheap to decode and the
        # radiologist would otherwise see nothing at all.
        assert should_skip_preview(open_tail_sec=1.0, decode_cost=3.0, max_lag_sec=3.0) is False

    def test_the_first_cycles_keep_the_preview(self) -> None:
        # decode_cost 0.0 means nothing has been timed yet — never guess.
        assert should_skip_preview(open_tail_sec=19.0, decode_cost=0.0, max_lag_sec=3.0) is False

    def test_zero_turns_the_skip_off(self) -> None:
        assert should_skip_preview(open_tail_sec=19.0, decode_cost=8.0, max_lag_sec=0.0) is False

    def test_exactly_at_the_threshold_is_not_yet_behind(self) -> None:
        assert should_skip_preview(open_tail_sec=3.0, decode_cost=3.0, max_lag_sec=3.0) is False


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


# ---------------------------------------------------------------------------
# Confidence-targeted polish
#
# This runs once, after recording stops. It replaced the old full re-transcribe,
# so it is the only thing standing between a shaky live guess and the report the
# radiologist signs. Two rules matter: spend the re-decode only where confidence
# was low, and never leave the last words of a dictation undecoded.
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402

from src.dictation.asr import AsrResult, AsrSegment, Word  # noqa: E402
from src.dictation.stream.ledger import ChunkLedger  # noqa: E402
from src.dictation.stream.segmenter import Chunk  # noqa: E402

SR = 16000


def _result(text: str, confidence: float) -> AsrResult:
    """One-segment result whose mean word confidence is exactly *confidence*."""
    word = Word(text=text, start=0.0, end=1.0, confidence=confidence)
    return AsrResult(text=text, segments=(AsrSegment(text, 0.0, 1.0, (word,)),))


class _RecordingEngine:
    """Returns queued results and remembers every clip length it was given."""

    def __init__(self, results):
        self._results = list(results)
        self.clip_lengths = []

    def transcribe(self, audio, ctx):
        self.clip_lengths.append(len(audio))
        return self._results.pop(0)


class _Recorder:
    def __init__(self):
        self.calls = []

    def emit(self, *args):
        self.calls.append(args)


def _polish_worker(monkeypatch, ledger, engine):
    monkeypatch.setattr(transcribe_worker, "get_custom_prompt_suffix", lambda: "")
    worker = object.__new__(LiveTranscribeWorker)
    worker.language = "en"
    worker.vad_enabled = True
    worker.pause_threshold = 2.5
    worker.silence_rms_floor = 0.002
    # Normally injected by the caller from settings (live_beam_size /
    # final_beam_size / polish_confidence_ceiling); __init__ is bypassed here.
    worker.live_beam_size = 2
    worker.final_beam_size = 5
    worker.polish_confidence_ceiling = 0.75
    worker._ledger = ledger
    worker._decode_sec_total = 0.0
    worker._last_emitted = ""
    worker.partial = _Recorder()
    worker.segments = _Recorder()
    return worker


class TestConfidenceTargetedPolish:
    def test_only_the_unsure_chunk_is_redecoded(self, monkeypatch) -> None:
        # The whole point of the gate: a chunk the model was sure about must
        # not be spent on again, and its text must survive untouched.
        ledger = ChunkLedger()
        ledger.commit(Chunk(0, SR, closed=True), "sure text", 0.95)
        ledger.commit(Chunk(SR, 2 * SR, closed=True), "shaky text", 0.40)
        engine = _RecordingEngine([_result("repaired text", 0.9)])
        worker = _polish_worker(monkeypatch, ledger, engine)
        audio = np.zeros(2 * SR, dtype=np.float32)  # silent tail -> no tail decode

        worker._run_confidence_targeted_polish(engine, audio, SR)

        texts = [c.text for c in ledger.committed]
        assert texts == ["sure text", "repaired text"]
        assert engine.clip_lengths == [SR]  # exactly one re-decode, the shaky chunk

    def test_audio_left_open_when_recording_stopped_still_gets_decoded(self, monkeypatch) -> None:
        # Without this the final second or two of every dictation would be lost:
        # it never closed into a chunk, so the live pass never froze it.
        ledger = ChunkLedger()
        ledger.commit(Chunk(0, SR, closed=True), "first part", 0.95)
        engine = _RecordingEngine([_result("last words", 0.9)])
        worker = _polish_worker(monkeypatch, ledger, engine)
        audio = np.concatenate([
            np.zeros(SR, dtype=np.float32),
            np.full(SR, 0.2, dtype=np.float32),  # loud enough to beat the floor
        ])

        worker._run_confidence_targeted_polish(engine, audio, SR)

        assert [c.text for c in ledger.committed] == ["first part", "last words"]
        assert ledger.open_start_sample == 2 * SR

    def test_a_silent_open_tail_is_not_decoded(self, monkeypatch) -> None:
        ledger = ChunkLedger()
        ledger.commit(Chunk(0, SR, closed=True), "first part", 0.95)
        engine = _RecordingEngine([])
        worker = _polish_worker(monkeypatch, ledger, engine)
        audio = np.zeros(2 * SR, dtype=np.float32)

        worker._run_confidence_targeted_polish(engine, audio, SR)

        assert engine.clip_lengths == []
        assert [c.text for c in ledger.committed] == ["first part"]

    def test_a_failed_redecode_keeps_the_original_text(self, monkeypatch) -> None:
        # A crashing engine on the polish pass must not blank a chunk the
        # radiologist already has on screen.
        class _BrokenEngine:
            def transcribe(self, audio, ctx):
                raise RuntimeError("engine died")

        ledger = ChunkLedger()
        ledger.commit(Chunk(0, SR, closed=True), "shaky text", 0.40)
        worker = _polish_worker(monkeypatch, ledger, _BrokenEngine())
        audio = np.zeros(SR, dtype=np.float32)

        worker._run_confidence_targeted_polish(_BrokenEngine(), audio, SR)

        assert [c.text for c in ledger.committed] == ["shaky text"]

    def test_a_chunk_with_no_confidence_signal_is_left_alone(self, monkeypatch) -> None:
        # mean_confidence None means "this engine gave no word timestamps", not
        # "the model was unsure" — re-decoding on that would defeat the gate.
        ledger = ChunkLedger()
        ledger.commit(Chunk(0, SR, closed=True), "unknown confidence", None)
        engine = _RecordingEngine([])
        worker = _polish_worker(monkeypatch, ledger, engine)
        audio = np.zeros(SR, dtype=np.float32)

        worker._run_confidence_targeted_polish(engine, audio, SR)

        assert engine.clip_lengths == []
        assert [c.text for c in ledger.committed] == ["unknown confidence"]
