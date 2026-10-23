"""Loudness helpers shared by the steps that follow the voice (expander, agc)."""

from __future__ import annotations

import numpy as np

_FLOOR_DB = -120.0


def frame_db(power: np.ndarray, window: int) -> float:
    """Level of one frame in dBFS (full-scale sine ~ -3). The window halves the
    mean power, hence the factor 2; the one-sided spectrum counts twice."""
    mean_square = 2.0 * 2.0 * float(power.sum()) / (window * window)
    return max(_FLOOR_DB, 10.0 * np.log10(mean_square + 1e-12))


class PeakFollower:
    """Follows the main voice's level: jumps up at once, falls slowly, so a
    pause or a quiet word never drags the voice level down with it."""

    def __init__(self, release_db_per_sec: float, frames_per_sec: float) -> None:
        self._fall = release_db_per_sec / frames_per_sec
        self.level: float | None = None

    def update(self, db: float) -> float:
        if self.level is None:
            self.level = db
        else:
            self.level = max(db, self.level - self._fall)
        return self.level
