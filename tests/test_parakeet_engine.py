"""Parakeet runs on half the cores, and says it is cheap and greedy."""

import sys
import types

from src.dictation.asr.engines import parakeet_engine as pe
from src.dictation.stream.policy import is_cheap


def test_loads_with_half_the_cores(monkeypatch):
    """Every core is fastest only when idle: two busy cores elsewhere made
    4 threads twice as slow as 2 on a 4-core machine."""
    seen = {}

    class _Model:
        def with_timestamps(self):
            return self

    def load_model(name, path, quantization=None, sess_options=None):
        seen["threads"] = sess_options.intra_op_num_threads
        return _Model()

    monkeypatch.setitem(sys.modules, "onnx_asr", types.SimpleNamespace(load_model=load_model))
    monkeypatch.setattr(pe.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(pe, "_MODELS", {})
    pe.ParakeetEngine().preload()
    assert seen["threads"] == 4


def test_never_zero_threads(monkeypatch):
    monkeypatch.setattr(pe.os, "cpu_count", lambda: 1)
    assert pe.default_cpu_threads() == 1


def test_declares_itself_cheap():
    assert is_cheap(pe.ParakeetEngine().capabilities())
