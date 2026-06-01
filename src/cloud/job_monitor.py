"""Background poller that advances cloud training jobs from the UI thread.

Runs on a QThread and periodically asks :class:`SyncManager` to poll in-flight
jobs. When a job finishes and its model is downloaded + registered, it emits
``model_available`` so the main window can offer to activate it. All heavy work
happens inside ``SyncManager``; this class only handles timing and signalling.

The monitor is inert unless cloud training is enabled, so it is always safe to
start at app launch.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)

# Poll interval — training takes minutes, so a slow cadence is plenty and keeps
# API usage minimal.
_POLL_INTERVAL_SEC = 60.0


class CloudJobMonitor(QObject):
    """Polls Lightning AI jobs and signals when a fine-tuned model is ready."""

    model_available = Signal(str)   # newly-registered model version
    status_changed = Signal(str)    # human-readable status line for the UI

    def __init__(self) -> None:
        super().__init__()
        self._keep_running = True

    def stop(self) -> None:
        self._keep_running = False

    def run(self) -> None:
        import time
        from src.cloud.sync_manager import SyncManager

        logger.info("CloudJobMonitor started")
        try:
            sync = SyncManager()
        except Exception as exc:
            logger.warning("CloudJobMonitor could not initialise: %s", exc)
            return

        while self._keep_running:
            try:
                if SyncManager._enabled():
                    # Kick off training if the threshold has been reached.
                    sync.maybe_start_training()
                    # Advance any in-flight jobs.
                    for version in sync.poll_and_collect():
                        self.model_available.emit(version)
            except Exception as exc:  # never let the monitor thread die
                logger.warning("CloudJobMonitor cycle error: %s", exc)

            # Sleep in short slices so stop() is responsive.
            slept = 0.0
            while slept < _POLL_INTERVAL_SEC and self._keep_running:
                time.sleep(1.0)
                slept += 1.0
        logger.info("CloudJobMonitor stopped")
