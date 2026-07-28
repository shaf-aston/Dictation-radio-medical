"""Which status message wins, and what a timed one reverts to.

Pure — no Qt — so the ordering rule can be tested. The rule matters because a
short message ("Applied 3 corrections", a clipping warning) schedules a revert a
few seconds later, and that revert used to fire blindly: a tip posted just
before Stop would land in the middle of finalising and claim the app was idle
while it was still working.
"""

from __future__ import annotations

READY = ("Ready", "idle")


class StatusTrack:
    """Counts status messages so a stale revert can recognise itself."""

    def __init__(self) -> None:
        self.generation = 0
        # The last thing the transcription worker said it was doing.
        self.last_progress: tuple[str, str] = ("Recording...", "rec")

    def show(self) -> int:
        """Register a new message and return its generation."""
        self.generation += 1
        return self.generation

    def revert(self, generation: int, dictating: bool) -> tuple[str, str] | None:
        """The message a timed status falls back to, or None if it is stale."""
        if generation != self.generation:
            return None
        return self.last_progress if dictating else READY
