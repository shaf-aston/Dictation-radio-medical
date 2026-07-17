"""Tests for background pre-warming (src/dictation/warmup.py).

These assert *behaviour that matters*, not that a function ran:

* warm-up actually leaves the slow SymSpell index built, so the first live
  chunk does not pay the ~1.3 s build — the whole point of the module;
* a warmer that raises is swallowed, so a broken optional dependency can never
  stop the app from starting;
* the async entry point warms exactly once per process.
"""

from __future__ import annotations

import pytest

from src.core import perf
from src.dictation import warmup
from src.medical import medical_dict


@pytest.fixture(autouse=True)
def _reset_state():
    perf.reset()
    warmup.reset()
    yield
    perf.reset()
    warmup.reset()


def _symspell_is_built() -> bool:
    """True iff the SymSpell singleton is already constructed (no build needed)."""
    return medical_dict._SYMSPELL is not None


class TestWarmUpBuildsTheSlowIndex:
    def test_symspell_is_built_after_warm_up(self) -> None:
        # This is the payoff: after warm-up the index exists, so the first live
        # postprocess call reuses it instead of building it on the hot path.
        if medical_dict.get_symspell() is None:
            pytest.skip("symspellpy not installed in this environment")
        medical_dict._SYMSPELL = None  # force a cold state
        assert not _symspell_is_built()

        warmup.warm_up(warmup.postprocess_warmers())

        assert _symspell_is_built()

    def test_warm_up_records_timings_for_each_warmer(self) -> None:
        warmup.warm_up([("demo_warmer", lambda: None)])

        assert "warmup.demo_warmer" in perf.snapshot()


class TestWarmUpNeverBreaksStartup:
    def test_a_failing_warmer_is_swallowed(self) -> None:
        def _boom() -> None:
            raise RuntimeError("optional dep missing")

        # Must not raise: warm-up is an optimisation, never a hard dependency.
        warmup.warm_up([("boom", _boom), ("ok", lambda: None)])

        # The good warmer after the failing one still ran and was timed.
        assert "warmup.ok" in perf.snapshot()


class TestAsyncWarmUpIsOnceOnly:
    def test_second_async_call_is_a_noop(self) -> None:
        calls: list[int] = []
        warmer = [("count", lambda: calls.append(1))]

        first = warmup.warm_up_async(warmer)
        second = warmup.warm_up_async(warmer)

        assert first is not None
        assert second is None  # already started this process
        first.join(timeout=5)
        assert calls == [1]


class TestTranscriberWarmerIsLabelled:
    def test_label_reflects_the_model(self) -> None:
        name, _ = warmup.transcriber_warmer("small")
        assert name == "whisper_model[small]"
