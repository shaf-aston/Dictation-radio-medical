"""Regression tests for the dictation performance/quality fixes.

Covers three contracts:
* process-wide Whisper model cache (``transcriber._MODEL_CACHE``),
* ``Transcriber.transcribe(temperature=...)`` forwarding (explicit value vs
  the default 5-step fallback ladder),
* ``WindowState`` paragraph preservation — frozen segments separated by a
  pause >= ``pause_threshold`` join with a newline, not a space.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from runtime_stubs import install_test_runtime_stubs

install_test_runtime_stubs()

import src.dictation.transcriber as transcriber_mod  # noqa: E402
from src.dictation.text_diff import trim_committed_tail  # noqa: E402
from src.dictation.transcriber import Transcriber  # noqa: E402
from src.dictation.window_state import WindowState  # noqa: E402

SR = 16000


def segment(start: float, end: float, text: str) -> dict:
    """Build a transcription segment payload."""
    return {"start": start, "end": end, "text": text}


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


class TestWindowStateParagraphs:
    """Frozen segments keep the paragraph break a dictation pause implies."""

    @staticmethod
    def make_state() -> WindowState:
        return WindowState(window_sec=25, commit_lag_sec=8, pause_threshold=2.5)

    def test_advance_commit_newline_on_long_pause(self) -> None:
        state = self.make_state()
        segments = [
            segment(0.0, 3.0, "Findings are normal."),
            segment(6.0, 9.0, "Impression follows."),  # 3.0s gap >= 2.5
        ]
        state.advance_commit(segments, chunk_start_sec=0.0, total_sec=20.0, sr=SR)
        assert state.committed_text == "Findings are normal.\nImpression follows."

    def test_advance_commit_space_on_short_gap(self) -> None:
        state = self.make_state()
        segments = [
            segment(0.0, 3.0, "Findings are normal."),
            segment(3.5, 9.0, "Impression follows."),  # 0.5s gap < 2.5
        ]
        state.advance_commit(segments, chunk_start_sec=0.0, total_sec=20.0, sr=SR)
        assert state.committed_text == "Findings are normal. Impression follows."

    def test_trim_committed_tail_keeps_newline_at_dedup_boundary(self) -> None:
        """Overlap dedup must not collapse a paragraph newline to a space."""
        committed = "Lungs clear. No effusion."
        addition = "No effusion.\nIMPRESSION: Normal chest."
        assert (
            trim_committed_tail(committed, addition)
            == "Lungs clear. No effusion.\nIMPRESSION: Normal chest."
        )

    def test_trim_committed_tail_keeps_leading_newline_without_overlap(self) -> None:
        committed = "Lungs clear."
        assert (
            trim_committed_tail(committed, "\nIMPRESSION: Normal chest.")
            == "Lungs clear.\nIMPRESSION: Normal chest."
        )

    def test_maybe_bootstrap_newline_on_long_pause(self) -> None:
        state = self.make_state()
        state.record_segments(
            [
                segment(0.0, 3.0, "Findings are normal."),
                segment(6.0, 9.0, "Impression follows."),  # 3.0s gap >= 2.5
            ],
            chunk_start_sec=0.0,
        )
        state.maybe_bootstrap(new_chunk_start_sec=10.0, sr=SR)
        assert state.committed_text == "Findings are normal.\nImpression follows."

    def test_newline_survives_across_commit_batches(self) -> None:
        """A pause spanning two advance_commit calls still gets its newline."""
        state = self.make_state()
        # Cycle N: only segment A is behind the frontier (safe_abs = 12-8 = 4).
        state.advance_commit(
            [segment(7.0, 10.0, "Findings are normal.")],
            chunk_start_sec=0.0, total_sec=18.0, sr=SR,
        )
        assert state.committed_text == "Findings are normal."
        # Cycle N+1: segment B (3s pause after A) commits in its own batch.
        state.advance_commit(
            [segment(13.0, 16.0, "Impression follows.")],
            chunk_start_sec=0.0, total_sec=25.0, sr=SR,
        )
        assert state.committed_text == "Findings are normal.\nImpression follows."

    def test_space_across_commit_batches_on_short_gap(self) -> None:
        state = self.make_state()
        state.advance_commit(
            [segment(7.0, 10.0, "Findings are normal.")],
            chunk_start_sec=0.0, total_sec=18.0, sr=SR,
        )
        state.advance_commit(
            [segment(10.5, 13.0, "No effusion.")],
            chunk_start_sec=0.0, total_sec=22.0, sr=SR,
        )
        assert state.committed_text == "Findings are normal. No effusion."

    def test_maybe_bootstrap_space_on_short_gap(self) -> None:
        state = self.make_state()
        state.record_segments(
            [
                segment(0.0, 3.0, "Findings are normal."),
                segment(3.5, 9.0, "Impression follows."),  # 0.5s gap < 2.5
            ],
            chunk_start_sec=0.0,
        )
        state.maybe_bootstrap(new_chunk_start_sec=10.0, sr=SR)
        assert state.committed_text == "Findings are normal. Impression follows."
