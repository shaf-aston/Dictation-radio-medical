"""Tests for the AsrEngine swap-seam (src/dictation/asr/).

No model, no audio — same stub pattern as test_dictation_perf_fixes.py: a fake
faster_whisper module is injected so FasterWhisperEngine can be exercised
without the real (heavy) dependency. These lock down the contract every
future engine (M3) and every caller (worker.py, web_app.py, the eval harness)
depends on: confidence is either a real mean or an honest None, and every
kwarg the port promises to forward actually reaches the underlying model.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from runtime_stubs import install_test_runtime_stubs

install_test_runtime_stubs()

import src.dictation.transcriber as transcriber_mod  # noqa: E402
from src.dictation.asr import (  # noqa: E402
    AsrResult,
    AsrSegment,
    TranscribeContext,
    Word,
    create_engine,
)
from src.dictation.asr.engines.faster_whisper_engine import FasterWhisperEngine  # noqa: E402

SR = 16000


# ---------------------------------------------------------------------------
# types.py
# ---------------------------------------------------------------------------

class TestAsrSegmentConfidence:
    def test_mean_of_word_confidences(self):
        seg = AsrSegment(text="ok", start=0.0, end=1.0, words=(
            Word("ok", 0.0, 0.5, confidence=0.8),
            Word("ok", 0.5, 1.0, confidence=0.4),
        ))
        assert seg.confidence == pytest.approx(0.6)

    def test_none_when_engine_gave_no_words(self):
        # Distinct from 0.0/1.0 — "unknown", not "certain" or "worthless".
        seg = AsrSegment(text="ok", start=0.0, end=1.0)
        assert seg.confidence is None

    def test_as_dict_matches_legacy_shape(self):
        seg = AsrSegment(text="hi", start=1.0, end=2.0)
        assert seg.as_dict() == {"start": 1.0, "end": 2.0, "text": "hi"}


class TestAsrResult:
    def test_segments_as_dicts(self):
        result = AsrResult(text="hi there", segments=(
            AsrSegment(text="hi", start=0.0, end=0.5),
            AsrSegment(text="there", start=0.5, end=1.0),
        ))
        assert result.segments_as_dicts() == [
            {"start": 0.0, "end": 0.5, "text": "hi"},
            {"start": 0.5, "end": 1.0, "text": "there"},
        ]


# ---------------------------------------------------------------------------
# factory.py
# ---------------------------------------------------------------------------

class TestFactory:
    def test_default_engine_is_faster_whisper(self):
        engine = create_engine(model_size="base", device="cpu")
        assert isinstance(engine, FasterWhisperEngine)

    def test_unknown_engine_name_raises(self):
        with pytest.raises(ValueError, match="Unknown ASR engine"):
            create_engine("not-a-real-engine")


# ---------------------------------------------------------------------------
# FasterWhisperEngine — behaviour against a fake WhisperModel
# ---------------------------------------------------------------------------

class _FakeWord:
    def __init__(self, word, start, end, probability):
        self.word, self.start, self.end, self.probability = word, start, end, probability


class _FakeSegment:
    def __init__(self, start, end, text, words=None):
        self.start, self.end, self.text = start, end, text
        self.words = words
        self.avg_logprob = -0.1
        self.no_speech_prob = 0.0


class _FakeWhisperModel:
    last_kwargs: dict = {}

    def __init__(self, *args, **kwargs) -> None:
        pass

    def transcribe(self, audio, **kwargs):
        type(self).last_kwargs = kwargs
        segs = [
            _FakeSegment(0.0, 1.0, "left kidney", words=[
                _FakeWord("left", 0.0, 0.5, 0.9),
                _FakeWord("kidney", 0.5, 1.0, 0.7),
            ]),
        ]
        return iter(segs), types.SimpleNamespace(language="en")


@pytest.fixture()
def fake_whisper(monkeypatch: pytest.MonkeyPatch):
    fake_module = types.ModuleType("faster_whisper")
    fake_module.WhisperModel = _FakeWhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    _FakeWhisperModel.last_kwargs = {}
    transcriber_mod._MODEL_CACHE.clear()
    yield _FakeWhisperModel
    transcriber_mod._MODEL_CACHE.clear()


class TestFasterWhisperEngine:
    def test_capabilities_advertise_word_confidence_and_hotwords(self, fake_whisper):
        engine = FasterWhisperEngine(model_size="base", device="cpu")
        caps = engine.capabilities()
        assert caps.word_confidence is True
        assert caps.hotwords is True

    def test_transcribe_returns_words_with_confidence(self, fake_whisper):
        engine = FasterWhisperEngine(model_size="base", device="cpu")
        result = engine.transcribe(
            np.zeros(SR, dtype=np.float32),
            TranscribeContext(want_word_confidence=True),
        )
        assert result.text == "left kidney"
        assert len(result.segments) == 1
        seg = result.segments[0]
        assert [w.text for w in seg.words] == ["left", "kidney"]
        assert seg.confidence == pytest.approx(0.8)

    def test_want_word_confidence_forwards_to_model(self, fake_whisper):
        engine = FasterWhisperEngine(model_size="base", device="cpu")
        engine.transcribe(
            np.zeros(SR, dtype=np.float32),
            TranscribeContext(want_word_confidence=True),
        )
        assert fake_whisper.last_kwargs["word_timestamps"] is True

    def test_default_context_does_not_request_word_timestamps(self, fake_whisper):
        # M1 must not change default behaviour: nothing downstream reads
        # .words yet, so the extra decode cost stays opt-in.
        engine = FasterWhisperEngine(model_size="base", device="cpu")
        engine.transcribe(np.zeros(SR, dtype=np.float32), TranscribeContext())
        assert fake_whisper.last_kwargs["word_timestamps"] is False

    def test_hotwords_forwarded_as_space_joined_string(self, fake_whisper):
        engine = FasterWhisperEngine(model_size="base", device="cpu")
        engine.transcribe(
            np.zeros(SR, dtype=np.float32),
            TranscribeContext(hotwords=["hydronephrosis", "pneumothorax"]),
        )
        assert fake_whisper.last_kwargs["hotwords"] == "hydronephrosis pneumothorax"

    def test_no_hotwords_key_when_none_given(self, fake_whisper):
        engine = FasterWhisperEngine(model_size="base", device="cpu")
        engine.transcribe(np.zeros(SR, dtype=np.float32), TranscribeContext())
        assert "hotwords" not in fake_whisper.last_kwargs

    def test_compute_type_reflects_the_loaded_transcriber(self, fake_whisper):
        engine = FasterWhisperEngine(model_size="base", device="cpu")
        assert engine.compute_type is None
        engine.preload()
        assert engine.compute_type is not None
