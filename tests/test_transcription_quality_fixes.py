"""Regressions for the perf/quality audit fixes (VAD, artifacts, hallucination
regex, boundary-commit dedup, final-pass config)."""

from __future__ import annotations

from src.dictation.transcriber import _is_hallucination
from src.dictation.postprocess.text_utils import normalize_spaces
from src.dictation.postprocess import postprocess_transcript
from src.dictation.worker import LiveTranscribeWorker


class _Seg:
    """Minimal faster-whisper segment stand-in for _is_hallucination()."""

    def __init__(self, start=0.0, end=1.0, avg_logprob=-0.2, no_speech_prob=0.1):
        self.start = start
        self.end = end
        self.avg_logprob = avg_logprob
        self.no_speech_prob = no_speech_prob


class TestHallucinationRegexBounding:
    def test_real_segments_starting_with_short_phrase_kept(self):
        # Previously deleted: "you" was prefix-matched.
        for text in (
            "your report shows a displaced fracture",
            "young patient with productive cough",
            "goodbye reflex intact on exam",
            "you are seeing early consolidation",
        ):
            assert not _is_hallucination(text, _Seg()), text

    def test_whole_segment_short_phrase_still_filtered(self):
        for text in ("you", "you.", "goodbye", "bye bye"):
            assert _is_hallucination(text, _Seg()), text

    def test_long_outro_phrase_still_prefix_filtered(self):
        assert _is_hallucination("thank you for watching this clip", _Seg())


class TestPunctuationRunArtifacts:
    def test_collapses_hallucinated_runs(self):
        assert normalize_spaces("findings , , . . , suggest") == "findings, suggest"
        assert normalize_spaces("normal . . heart") == "normal. heart"

    def test_spoken_punctuation_single_marks_survive(self):
        # The spoken-punctuation stage emits " : " / " . "; must not be eaten.
        assert postprocess_transcript("findings colon impression") == "Findings: impression"
        assert postprocess_transcript("end period") == "End."

    def test_legit_sentence_and_numeric_punctuation_preserved(self):
        assert normalize_spaces("The heart is normal. No effusion.") == "The heart is normal. No effusion."
        assert normalize_spaces("measures 3.5 cm, ratio 2:1") == "measures 3.5 cm, ratio 2:1"


class TestSilenceRms:
    def test_rms_zero_on_silence(self):
        import numpy as np
        assert LiveTranscribeWorker._rms(np.zeros(1000, dtype="float32")) == 0.0

    def test_rms_positive_on_signal(self):
        import numpy as np
        sig = (np.ones(1000, dtype="float32") * 0.1)
        assert abs(LiveTranscribeWorker._rms(sig) - 0.1) < 1e-6
