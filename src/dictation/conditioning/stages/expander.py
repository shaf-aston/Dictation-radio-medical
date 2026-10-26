"""Expander: turns down frames far below the main voice (a colleague, a fan)."""

from __future__ import annotations

import numpy as np

from src.dictation.conditioning.stages.level import PeakFollower, frame_db


class Expander:
    def __init__(
        self, below_voice_db: float, max_cut_db: float,
        voice_release_db_per_sec: float, window: int, frames_per_sec: float,
    ) -> None:
        self._below = below_voice_db
        self._max_cut = max_cut_db
        self._window = window
        self._voice = PeakFollower(voice_release_db_per_sec, frames_per_sec)

    def gain(self, power: np.ndarray) -> float:
        db = frame_db(power, self._window)
        voice = self._voice.update(db)
        # Eases in over the dB past the line and stops at max_cut_db: it only
        # turns down, never off, so a quiet word's start is dimmed, not lost.
        cut = min(self._max_cut, max(0.0, voice - self._below - db))
        return 10.0 ** (-cut / 20.0)
