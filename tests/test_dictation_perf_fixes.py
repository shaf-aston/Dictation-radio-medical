"""Regression tests for the dictation performance/quality fixes.

Covers two contracts:
* process-wide Whisper model cache (``transcriber._MODEL_CACHE``),
* ``Transcriber.transcribe(temperature=...)`` forwarding (explicit value vs
  the default 5-step fallback ladder).

(Paragraph-join / boundary-dedup coverage moved to tests/test_stream.py's
ChunkLedger tests — window_state.py and text_diff.py, the sliding-window
re-decode machinery this used to exercise, were superseded by the
chunk-once ledger and deleted.)
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from runtime_stubs import install_test_runtime_stubs

install_test_runtime_stubs()

import src.dictation.transcriber as transcriber_mod  # noqa: E402
from src.dictation.transcriber import Transcriber  # noqa: E402

SR = 16000


# ---------------------------------------------------------------------------
# Counting WhisperModel stub — injected as a fake faster_whisper module so
# the heavy real dependency is never imported.
# ---------------------------------------------------------------------------


class _FakeWhisperModel:
    """Counts constructions and captures transcribe() kwargs."""

    constructed: list = []  # (model_ref, device, compute_type) per construction
    init_kwargs: list = []  # extra kwargs (download_root, cpu_threads, ...) per construction
    last_transcribe_kwargs: dict = {}

    def __init__(self, model_ref, device="auto", compute_type=None, **kwargs) -> None:
        type(self).constructed.append((model_ref, device, compute_type))
        type(self).init_kwargs.append(kwargs)

    def transcribe(self, audio, **kwargs):
        type(self).last_transcribe_kwargs = kwargs
        return iter([]), types.SimpleNamespace(language="en")


@pytest.fixture()
def fake_whisper(monkeypatch: pytest.MonkeyPatch):
    """Install the counting stub and reset the process-wide model cache."""
    fake_module = types.ModuleType("faster_whisper")
    fake_module.WhisperModel = _FakeWhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    _FakeWhisperModel.constructed = []
    _FakeWhisperModel.init_kwargs = []
    _FakeWhisperModel.last_transcribe_kwargs = {}
    transcriber_mod._MODEL_CACHE.clear()
    yield _FakeWhisperModel
    transcriber_mod._MODEL_CACHE.clear()


class TestModelCache:
    """WhisperModel instances are cached process-wide, not per Transcriber."""

    def test_same_config_shares_one_model(self, fake_whisper) -> None:
        a = Transcriber(model_size="base", device="cpu")
        b = Transcriber(model_size="base", device="cpu")
        a.preload()
        b.preload()
        assert len(fake_whisper.constructed) == 1
        assert a._model is b._model

    def test_different_model_size_constructs_again(self, fake_whisper) -> None:
        a = Transcriber(model_size="base", device="cpu")
        b = Transcriber(model_size="small", device="cpu")
        a.preload()
        b.preload()
        assert len(fake_whisper.constructed) == 2
        assert a._model is not b._model

    def test_preload_is_idempotent(self, fake_whisper) -> None:
        t = Transcriber(model_size="base", device="cpu")
        t.preload()
        t.preload()
        assert len(fake_whisper.constructed) == 1

    def test_download_root_is_the_project_whisper_cache(self, fake_whisper) -> None:
        """Stock models must download into data/cache/whisper/, never the
        hidden per-user HuggingFace cache."""
        from src.features.file_manager import whisper_cache_dir

        t = Transcriber(model_size="base", device="cpu")
        t.preload()
        assert fake_whisper.init_kwargs[0]["download_root"] == str(whisper_cache_dir())


class TestTemperatureForwarding:
    """transcribe() forwards an explicit temperature or the fallback ladder."""

    def test_explicit_temperature_forwarded(self, fake_whisper) -> None:
        t = Transcriber(model_size="base", device="cpu")
        t.transcribe(np.zeros(SR, dtype=np.float32), temperature=0.0)
        assert fake_whisper.last_transcribe_kwargs["temperature"] == 0.0

    def test_default_uses_fallback_ladder(self, fake_whisper) -> None:
        t = Transcriber(model_size="base", device="cpu")
        t.transcribe(np.zeros(SR, dtype=np.float32))
        assert fake_whisper.last_transcribe_kwargs["temperature"] == [
            0.0, 0.2, 0.4, 0.6, 0.8,
        ]
