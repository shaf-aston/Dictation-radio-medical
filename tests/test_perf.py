"""Tests for the performance-tracking module (src/core/perf.py)."""

from __future__ import annotations

import threading

import pytest

from src.core import perf


@pytest.fixture(autouse=True)
def clean_perf():
    perf.reset()
    yield
    perf.reset()


class TestRecording:
    def test_stage_records_a_sample(self) -> None:
        with perf.stage("demo"):
            pass

        assert perf.snapshot()["demo"]["count"] == 1

    def test_stage_records_even_when_the_block_raises(self) -> None:
        # A stage that fails slowly still has to show up, or a slow failure path
        # stays invisible in the timings.
        with pytest.raises(ValueError):
            with perf.stage("boom"):
                raise ValueError("x")

        assert perf.snapshot()["boom"]["count"] == 1

    def test_timed_decorator_records_calls(self) -> None:
        @perf.timed("decorated")
        def work(value: int) -> int:
            return value * 2

        assert work(3) == 6
        assert perf.snapshot()["decorated"]["count"] == 1

    def test_empty_snapshot_and_summary_are_safe(self) -> None:
        assert perf.snapshot() == {}
        perf.log_summary()  # must not raise


class TestGauges:
    def test_set_gauge_is_readable(self) -> None:
        perf.set_gauge("stream.decode_ratio", 1.35)
        assert perf.gauges()["stream.decode_ratio"] == pytest.approx(1.35)

    def test_set_gauge_overwrites_not_accumulates(self) -> None:
        perf.set_gauge("g", 1.0)
        perf.set_gauge("g", 2.0)
        assert perf.gauges()["g"] == pytest.approx(2.0)

    def test_gauges_do_not_leak_into_stage_snapshot(self) -> None:
        perf.set_gauge("g", 1.0)
        assert perf.snapshot() == {}

    def test_reset_clears_gauges(self) -> None:
        perf.set_gauge("g", 1.0)
        perf.reset()
        assert perf.gauges() == {}


class TestStats:
    def test_stats_are_computed_over_recorded_samples(self) -> None:
        for value in (0.01, 0.02, 0.03, 0.04):
            perf.record("s", value)

        stats = perf.snapshot()["s"]
        assert stats["count"] == 4
        assert stats["mean_ms"] == pytest.approx(25.0)
        assert stats["max_ms"] == pytest.approx(40.0)
        assert stats["total_ms"] == pytest.approx(100.0)
        assert stats["p95_ms"] == pytest.approx(40.0)

    def test_samples_are_bounded(self) -> None:
        # Memory must not grow with session length.
        for _ in range(perf._MAX_SAMPLES + 50):
            perf.record("bounded", 0.001)

        assert perf.snapshot()["bounded"]["count"] == perf._MAX_SAMPLES

    def test_snapshot_is_ordered_by_total_time_desc(self) -> None:
        perf.record("cheap", 0.001)
        perf.record("expensive", 1.0)

        assert list(perf.snapshot()) == ["expensive", "cheap"]

    def test_records_from_multiple_threads_are_all_kept(self) -> None:
        # The pipeline runs off the UI thread, so recording is concurrent.
        def worker() -> None:
            for _ in range(50):
                perf.record("concurrent", 0.001)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert perf.snapshot()["concurrent"]["count"] == 200
