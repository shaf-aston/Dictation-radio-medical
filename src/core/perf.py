"""Lightweight in-process performance tracking.

Purpose: make slowness *visible and provable* instead of guessed at. Every hot
stage (audio read, Whisper decode, each post-process stage, UI apply) wraps
itself in :func:`stage`; the timings accumulate into bounded rolling stats that
can be logged on demand or served to the web debug view.

Deliberately small and dependency-free:

* **stdlib only** — no external metrics library.
* **no telemetry** — nothing leaves the process, let alone the device. This
  keeps the CLAUDE.md "offline by default" invariant true by construction.
* **bounded memory** — each stage keeps at most :data:`_MAX_SAMPLES` recent
  durations in a ``deque``, so a long dictation session cannot grow it.

Usage::

    from src.core import perf

    with perf.stage("postprocess.terminology"):
        text = apply_terminology(text)

    @perf.timed("worker.read_audio")
    def _read_audio(self): ...

    perf.log_summary()          # one INFO line per stage, slowest first
    perf.snapshot()             # dict for the /api/debug/perf endpoint
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, TypeVar

logger = logging.getLogger(__name__)

# Rolling window per stage. 512 samples is ~an hour of live cycles at the
# fastest cadence — enough for a stable p95, small enough to be free.
_MAX_SAMPLES = 512

_lock = threading.Lock()
_samples: Dict[str, deque] = {}
# Point-in-time ratios/counts (e.g. stream.decode_ratio) — distinct from the
# rolling *_ms timing samples above, which would misrepresent a unitless
# ratio by scaling it x1000 as if it were seconds.
_gauges: Dict[str, float] = {}

F = TypeVar("F", bound=Callable[..., Any])


def record(name: str, seconds: float) -> None:
    """Record one timing sample for *name*."""
    with _lock:
        bucket = _samples.get(name)
        if bucket is None:
            bucket = _samples[name] = deque(maxlen=_MAX_SAMPLES)
        bucket.append(seconds)


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Time the enclosed block and record it under *name*.

    The sample is recorded even when the block raises, so a stage that fails
    slowly still shows up in the summary.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        record(name, time.perf_counter() - start)


def set_gauge(name: str, value: float) -> None:
    """Record the latest value of a point-in-time metric, e.g. one recording's
    ``stream.decode_ratio`` (decoded seconds / audio seconds). Overwrites any
    previous value for *name* — a gauge is "what is it now", not a history."""
    with _lock:
        _gauges[name] = value


def gauges() -> Dict[str, float]:
    """Current value of every gauge set via :func:`set_gauge`."""
    with _lock:
        return dict(_gauges)


def timed(name: str) -> Callable[[F], F]:
    """Decorator form of :func:`stage`."""

    def decorate(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with stage(name):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorate


def _percentile(sorted_vals: list[float], fraction: float) -> float:
    """Nearest-rank percentile of an already-sorted, non-empty list."""
    index = min(
        len(sorted_vals) - 1,
        max(0, round(fraction * len(sorted_vals) + 0.5) - 1),
    )
    return sorted_vals[index]


def snapshot() -> Dict[str, Dict[str, float]]:
    """Return ``{stage: {count, mean_ms, p95_ms, max_ms, total_ms}}``.

    Sorted slowest-total-first: the stage costing the most wall-clock over the
    session leads, which is the one worth optimising.
    """
    with _lock:
        raw = {name: list(vals) for name, vals in _samples.items()}

    stats: Dict[str, Dict[str, float]] = {}
    for name, vals in raw.items():
        if not vals:
            continue
        ordered = sorted(vals)
        total = sum(vals)
        stats[name] = {
            "count": len(vals),
            "mean_ms": round(total / len(vals) * 1000, 2),
            "p95_ms": round(_percentile(ordered, 0.95) * 1000, 2),
            "max_ms": round(ordered[-1] * 1000, 2),
            "total_ms": round(total * 1000, 2),
        }
    return dict(
        sorted(stats.items(), key=lambda kv: kv[1]["total_ms"], reverse=True)
    )


def log_summary(title: str = "perf") -> None:
    """Log one INFO line per stage, slowest total first, then every gauge.

    Gauges are logged too because ``stream.decode_ratio`` is the headline number for
    chunk-once streaming — leaving it out of the end-of-recording log meant the one
    metric worth reading was only ever visible at ``GET /api/debug/perf``.
    No-op when there is nothing to report.
    """
    stats = snapshot()
    current = gauges()
    if not stats and not current:
        return
    logger.info("--- %s (rolling, last %d samples/stage) ---", title, _MAX_SAMPLES)
    for name, s in stats.items():
        logger.info(
            "%-38s n=%-5d mean=%7.1fms  p95=%7.1fms  max=%7.1fms  total=%8.1fms",
            name, int(s["count"]), s["mean_ms"], s["p95_ms"],
            s["max_ms"], s["total_ms"],
        )
    for name, value in sorted(current.items()):
        logger.info("%-38s %.3f", name, value)


def reset() -> None:
    """Drop all samples and gauges (used between recordings and by tests)."""
    with _lock:
        _samples.clear()
        _gauges.clear()
