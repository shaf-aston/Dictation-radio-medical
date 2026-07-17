"""Off-thread post-processing for live dictation (desktop).

Why this exists
---------------
The live worker re-emits the whole transcript several times a second. Running
the ten-stage correction pipeline inline in the ``partial`` slot ran it on the
Qt **main thread** — so every cycle the UI stopped repainting and stopped
accepting input until terminology regexes and the fuzzy medical dictionary had
finished scanning the entire document. That is the freeze the user feels.

This moves the pipeline onto its own thread, with two properties that matter:

* **Coalescing.** Only the *latest* transcript is ever processed. If three
  cycles land while a pass is running, the two older ones are dropped — they
  are stale by definition (each emission supersedes the last), so processing
  them would burn CPU to produce text that is immediately overwritten.
* **Ordering.** Results carry the sequence number they were submitted with, so
  a late result can never overwrite a newer one.

Qt rules observed: this object never touches a widget. It emits a signal; the
UI thread owns every widget mutation.
"""

from __future__ import annotations

import logging
import threading
from typing import List, Optional, Tuple

from PySide6.QtCore import QObject, Signal, Slot

from src.core import perf
from src.dictation.postprocess.incremental import IncrementalPostprocessor

logger = logging.getLogger(__name__)


class PostprocessWorker(QObject):
    """Runs the post-process pipeline off the UI thread, latest-only.

    Lives on its own ``QThread``. :meth:`submit` is called from the UI thread;
    :attr:`processed` is delivered back on the UI thread by Qt's queued
    connection.
    """

    #: ``(processed_text, changes, seq)`` — emitted on the UI thread.
    processed = Signal(str, list, int)

    #: Internal: wakes the worker thread. Never connect from outside.
    _wake = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._incremental = IncrementalPostprocessor()
        self._lock = threading.Lock()
        self._pending: Optional[Tuple[str, int, str, str, int]] = None
        # Auto-connection: resolved as queued once this object is moved to its
        # own thread, so _drain always runs there, never on the caller's thread.
        self._wake.connect(self._drain)

    # -- UI thread ------------------------------------------------------

    def submit(
        self,
        text: str,
        committed_len: int,
        accent: str,
        cleanup_level: str,
        seq: int,
    ) -> None:
        """Queue *text* for processing, replacing any transcript still waiting."""
        with self._lock:
            self._pending = (text, committed_len, accent, cleanup_level, seq)
        self._wake.emit()

    # -- worker thread --------------------------------------------------

    @Slot()
    def _drain(self) -> None:
        """Process the newest pending transcript, if any."""
        with self._lock:
            job, self._pending = self._pending, None
        if job is None:
            # A superseded wake-up: the transcript it referred to was already
            # replaced and processed. Nothing to do.
            return

        text, committed_len, accent, cleanup_level, seq = job
        try:
            with perf.stage("postprocess.total"):
                self._incremental.configure(accent, cleanup_level)
                processed, changes = self._incremental.process(text, committed_len)
        except Exception as exc:
            # Never lose the user's words to a post-processing bug: fall back to
            # the raw transcript rather than dropping the emission.
            logger.warning("Post-processing failed, using raw text: %s", exc)
            processed, changes = text, []
        self.processed.emit(processed, list(changes), seq)

    @Slot()
    def reset(self) -> None:
        """Drop cached prefix state (new recording)."""
        with self._lock:
            self._pending = None
        self._incremental.reset()


def build_changes(existing_seen: set, changes: List[str]) -> List[str]:
    """Return the changes in *changes* not already in *existing_seen*, marking them.

    Shared by the desktop corrections banner: the same cumulative changes are
    re-reported every cycle, so membership is tracked in a set rather than
    re-scanning a list (which was quadratic per session).
    """
    fresh = [c for c in changes if c not in existing_seen]
    existing_seen.update(fresh)
    return fresh
