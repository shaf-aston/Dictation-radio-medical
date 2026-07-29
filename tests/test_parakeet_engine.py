"""Tests for the Parakeet AsrEngine (src/dictation/asr/engines/parakeet_engine.py).

No model, no network, no onnxruntime: a fake ``onnx_asr`` module stands in, the
same way test_asr_port.py fakes ``faster_whisper``. These lock down the four
things that can go wrong without anyone noticing — the factory resolving the
name, capabilities telling the truth about what this engine can and cannot do,
a missing package failing with an install hint instead of an ImportError at
startup, and empty audio never reaching the preprocessor.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from runtime_stubs import install_test_runtime_stubs

install_test_runtime_stubs()

from src.dictation.asr import TranscribeContext, create_engine  # noqa: E402
from src.dictation.asr.factory import ENGINE_NAMES, model_kwargs  # noqa: E402
from src.dictation.asr.engines import parakeet_engine as pk  # noqa: E402
from src.dictation.asr.engines.parakeet_engine import ParakeetEngine  # noqa: E402

SR = 16000


# ---------------------------------------------------------------------------
# A fake onnx-asr: records what it was asked for, returns fixed tokens
# ---------------------------------------------------------------------------

class _FakeResult:
    # Two words from five sub-word pieces, so the merge is actually exercised:
    # onnx-asr has already turned SentencePiece's marker into a leading space.
    text = "left kidney"
    tokens = [" le", "ft", " kid", "ne", "y"]
    timestamps = [0.0, 0.2, 0.4, 0.6, 0.8]
    logprobs = [0.0, 0.0, 0.0, 0.0, 0.0]  # exp(0) = probability 1.0


class _FakeModel:
    last_recognize: dict = {}

    def with_timestamps(self):
        return self

    def recognize(self, waveform, **kwargs):
        type(self).last_recognize = {"waveform": waveform, **kwargs}
        return _FakeResult()


@pytest.fixture()
def fake_onnx_asr(monkeypatch: pytest.MonkeyPatch):
    calls: dict = {}

    def load_model(model, path=None, **kwargs):
        calls.update({"model": model, "path": path, **kwargs})
        return _FakeModel()

    module = types.ModuleType("onnx_asr")
    module.load_model = load_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnx_asr", module)
    _FakeModel.last_recognize = {}
    yield calls


@pytest.fixture()
def no_onnx_asr(monkeypatch: pytest.MonkeyPatch):
    """Make ``import onnx_asr`` fail, as on a machine without the package."""
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) \
        else __builtins__.__import__

    def blocked(name, *args, **kwargs):
        if name == "onnx_asr":
            raise ImportError("No module named 'onnx_asr'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "onnx_asr", None)
    monkeypatch.setattr("builtins.__import__", blocked)
    yield


# ---------------------------------------------------------------------------
# factory.py
# ---------------------------------------------------------------------------

class TestFactory:
    def test_parakeet_resolves_by_name(self):
        assert isinstance(create_engine("parakeet"), ParakeetEngine)

    def test_name_is_listed_for_callers(self):
        assert "parakeet" in ENGINE_NAMES

    def test_model_kwarg_is_engine_specific(self):
        # The eval harness has one --model flag; the factory owns the mapping
        # onto each engine's own constructor argument name.
        assert model_kwargs("parakeet", "x") == {"model_name": "x"}
        assert model_kwargs("faster-whisper", "x") == {"model_size": "x"}

    def test_blank_model_selects_the_engine_default(self):
        assert model_kwargs("parakeet", "") == {}

    def test_unknown_engine_raises(self):
        with pytest.raises(ValueError, match="Unknown ASR engine"):
            model_kwargs("not-a-real-engine", "x")


# ---------------------------------------------------------------------------
# capabilities() — honesty about what this engine does not do
# ---------------------------------------------------------------------------

class TestCapabilities:
    def test_reports_word_confidence_and_no_hotwords(self):
        # A Transducer has no prompt slot, so hotword biasing is not available
        # and must not be advertised; word probabilities genuinely are.
        caps = ParakeetEngine().capabilities()
        assert caps.word_confidence is True
        assert caps.hotwords is False

    def test_capabilities_needs_no_model(self):
        # Checked before anything is loaded, so it cannot touch the network.
        assert ParakeetEngine(model_name="never-downloaded").capabilities()


# ---------------------------------------------------------------------------
# transcribe()
# ---------------------------------------------------------------------------

class TestTranscribe:
    def test_merges_pieces_into_words_with_confidence(self, fake_onnx_asr):
        result = ParakeetEngine().transcribe(
            np.zeros(SR, dtype=np.float32),
            TranscribeContext(want_word_confidence=True),
        )
        assert result.text == "left kidney"
        words = result.segments[0].words
        assert [w.text for w in words] == ["left", "kidney"]
        assert [w.confidence for w in words] == pytest.approx([1.0, 1.0])

    def test_words_line_up_with_the_text(self, fake_onnx_asr):
        # The post-processing confidence veto pairs confidences with
        # text.split() positionally and goes inert on a mismatch.
        result = ParakeetEngine().transcribe(
            np.zeros(SR, dtype=np.float32), TranscribeContext(),
        )
        words = [w.text for w in result.segments[0].words]
        assert words == result.text.split()

    def test_word_ends_at_the_next_word_start(self, fake_onnx_asr):
        result = ParakeetEngine().transcribe(
            np.zeros(SR, dtype=np.float32), TranscribeContext(),
        )
        first, second = result.segments[0].words
        assert first.start == 0.0
        assert first.end == second.start == pytest.approx(0.4)

    def test_want_word_confidence_false_still_works(self, fake_onnx_asr):
        # Parakeet gets its probabilities for free, so the flag changes
        # nothing — what matters is that asking either way cannot crash.
        result = ParakeetEngine().transcribe(
            np.zeros(SR, dtype=np.float32),
            TranscribeContext(want_word_confidence=False),
        )
        assert result.text == "left kidney"

    def test_empty_audio_returns_empty_without_loading_a_model(self, fake_onnx_asr):
        result = ParakeetEngine().transcribe(
            np.zeros(0, dtype=np.float32), TranscribeContext(),
        )
        assert result.text == ""
        assert result.segments == ()
        assert fake_onnx_asr == {}  # load_model was never called

    def test_blank_transcription_yields_no_segment(self, fake_onnx_asr, monkeypatch):
        monkeypatch.setattr(_FakeResult, "text", "   ")
        result = ParakeetEngine().transcribe(
            np.zeros(SR, dtype=np.float32), TranscribeContext(),
        )
        assert result.text == ""
        assert result.segments == ()

    def test_a_path_is_passed_through_for_onnx_asr_to_read(self, fake_onnx_asr):
        ParakeetEngine().transcribe("clip.wav", TranscribeContext())
        assert _FakeModel.last_recognize["waveform"] == "clip.wav"

    def test_non_float32_audio_is_converted(self, fake_onnx_asr):
        ParakeetEngine().transcribe(np.zeros(SR, dtype=np.float64), TranscribeContext())
        assert _FakeModel.last_recognize["waveform"].dtype == np.float32


class TestMisalignedTokens:
    def test_confidence_is_dropped_rather_than_misassigned(self, fake_onnx_asr, monkeypatch):
        # Zipping mismatched lists would silently label the wrong word as the
        # one the decoder doubted, which is worse than no confidence at all.
        monkeypatch.setattr(_FakeResult, "logprobs", [0.0, 0.0])
        result = ParakeetEngine().transcribe(
            np.zeros(SR, dtype=np.float32), TranscribeContext(),
        )
        assert result.text == "left kidney"
        assert result.segments[0].words == ()
        assert result.segments[0].confidence is None


# ---------------------------------------------------------------------------
# The package being absent
# ---------------------------------------------------------------------------

class TestPackageAbsent:
    def test_constructing_the_engine_needs_nothing_installed(self, no_onnx_asr):
        # Importing the factory must not drag onnxruntime into a core app that
        # never asks for this engine.
        assert ParakeetEngine() is not None

    def test_preload_fails_with_an_install_hint(self, no_onnx_asr):
        with pytest.raises(RuntimeError, match="pip install onnx-asr"):
            ParakeetEngine().preload()


# ---------------------------------------------------------------------------
# Model cache location
# ---------------------------------------------------------------------------

class TestModelCache:
    def test_downloads_into_the_project_cache(self, fake_onnx_asr):
        engine = ParakeetEngine()
        engine.preload()
        path = fake_onnx_asr["path"]
        assert path.parent.name == "onnx_asr"
        assert path.parent.parent.name == "cache"
        assert fake_onnx_asr["model"] == pk.DEFAULT_MODEL
        assert fake_onnx_asr["quantization"] == pk.DEFAULT_QUANTIZATION

    def test_quantisation_is_part_of_the_directory_name(self, fake_onnx_asr):
        # Two quantisations of one model are different downloads; sharing a
        # directory would make onnx-asr load whichever landed first.
        ParakeetEngine(quantization="int8").preload()
        int8_path = fake_onnx_asr["path"]
        ParakeetEngine(quantization=None).preload()
        assert fake_onnx_asr["path"] != int8_path

    def test_the_cache_directory_is_not_created_ahead_of_the_download(self, fake_onnx_asr):
        # onnx-asr reads an existing directory as "already downloaded, stay
        # offline", so pre-making it would break the very first run.
        ParakeetEngine().preload()
        assert not fake_onnx_asr["path"].exists()

    def test_model_is_loaded_once(self, fake_onnx_asr):
        engine = ParakeetEngine()
        engine.preload()
        first = engine._model
        engine.preload()
        assert engine._model is first
