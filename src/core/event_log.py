"""A short, in-memory diary of what the app just did: and how long it took.

Why this exists next to :mod:`src.core.perf`: perf answers *"what does this
stage cost on average?"*; this answers *"what happened, in what order, just
now?"*. Debugging a slow dictation needs both: the rolling mean says the
decode costs 1.4s, the diary says which decode blew out to 6s and what was
happening around it.

Deliberately the same shape as the rest of ``src/core``:

* **stdlib only**, no metrics or logging library.
* **nothing leaves the process**: no file, no socket, no telemetry. The
  offline-by-default invariant is untouched; the web app only ever serves this
  back to ``127.0.0.1`` on request.
* **bounded**: a ring buffer, so a long session cannot grow it.

Usage::

    from src.core import event_log

    event_log.emit("live", "socket open", model="tiny.en")

    with event_log.timed("live", "decode chunk", index=3):
        result = engine.transcribe(...)      # records ms= on the way out

    event_log.attach_to_logging()            # mirror ordinary log lines in too
    event_log.events(after=0)                # what the /api/debug/events serves
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Any, Deque, Dict, Iterator, List, Optional

from src.core import perf

# How many events are kept. One dictation cycle emits a handful, so a thousand
# is several minutes of live dictation: long enough to scroll back through the
# recording you just did, small enough to be free.
DEFAULT_CAPACITY = 1000

_lock = threading.Lock()
_events: Deque[Dict[str, Any]] = deque(maxlen=DEFAULT_CAPACITY)
_seq = 0
_handler: Optional[logging.Handler] = None
# What is in flight right now, keyed by id(): the console's "running now".
_running: Dict[int, Dict[str, Any]] = {}


def configure(capacity: int) -> None:
    """Resize the buffer, keeping the most recent events that still fit."""
    global _events
    if capacity < 1:
        raise ValueError("event log capacity must be at least 1")
    with _lock:
        _events = deque(_events, maxlen=capacity)


def emit(
    source: str,
    message: str,
    *,
    level: str = "info",
    ms: Optional[float] = None,
    **fields: Any,
) -> None:
    """Record one event.

    *source* is the subsystem ("live", "web", "asr"), *message* is a short
    human sentence, *ms* is a duration when the event is about something that
    took time, and *fields* are whatever numbers make the line readable.
    """
    global _seq
    with _lock:
        _seq += 1
        _events.append(
            {
                "seq": _seq,
                "t": time.time(),
                "level": level,
                "source": source,
                "message": message,
                "ms": None if ms is None else round(ms, 1),
                "fields": {k: v for k, v in fields.items() if v is not None},
            }
        )


@contextmanager
def timed(source: str, message: str, *, stage: Optional[str] = None, **fields: Any) -> Iterator[Dict[str, Any]]:
    """Time the enclosed block, then emit it with ``ms=``.

    Yields a plain dict: put anything you only learn *inside* the block (a
    word count, a result length) into it and it is emitted alongside. The
    event is emitted even when the block raises, marked ``level="error"``, so a
    stage that fails slowly still shows up in the diary.

    Pass *stage* to also feed :mod:`src.core.perf`, so the same measurement
    contributes to the rolling mean without being timed twice.
    """
    extra: Dict[str, Any] = {}
    start = time.perf_counter()
    level = "info"
    token = {"source": source, "message": message, "started": time.time()}
    with _lock:
        _running[id(token)] = token
    try:
        yield extra
    except BaseException:
        level = "error"
        raise
    finally:
        elapsed = time.perf_counter() - start
        with _lock:
            _running.pop(id(token), None)
        if stage:
            perf.record(stage, elapsed)
        emit(source, message, level=level, ms=elapsed * 1000, **{**fields, **extra})


def events(after: int = 0, limit: int = 500) -> List[Dict[str, Any]]:
    """Events with ``seq > after``, oldest first, at most *limit* of them.

    The caller passes back the last ``seq`` it saw, so polling never re-sends
    a line the developer panel has already printed.
    """
    with _lock:
        recent = [e for e in _events if e["seq"] > after]
    return recent[-limit:]


def running() -> List[Dict[str, Any]]:
    """Timed blocks still open, oldest first: what the app is doing right now."""
    with _lock:
        return sorted(_running.values(), key=lambda r: r["started"])


def latest_seq() -> int:
    """Sequence number of the newest event (0 when nothing has happened)."""
    with _lock:
        return _seq


def reset() -> None:
    """Drop every event (used between recordings and by tests)."""
    global _seq
    with _lock:
        _events.clear()
        _seq = 0


class _LoggingBridge(logging.Handler):
    """Mirrors ordinary log records into the diary.

    Without this the panel would show only the events this module was told
    about and miss the ``logger.warning`` that explains them: the two halves
    of the story would be in two different places.
    """

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102 - logging API
        try:
            message = record.getMessage()
        except Exception:  # a bad format string must never break logging
            message = record.msg if isinstance(record.msg, str) else "<unformattable>"
        # Marked, so the console can hide ordinary chatter (model loaded,
        # rules read) and show only what took time or went wrong.
        emit(record.name.split(".")[-1], message, level=record.levelname.lower(), mirrored=True)


def attach_to_logging(level: int = logging.INFO) -> None:
    """Start mirroring root log records into the diary. Idempotent."""
    global _handler
    if _handler is not None:
        return
    _handler = _LoggingBridge(level=level)
    logging.getLogger().addHandler(_handler)


def detach_from_logging() -> None:
    """Stop mirroring log records (used by tests)."""
    global _handler
    if _handler is None:
        return
    logging.getLogger().removeHandler(_handler)
    _handler = None
