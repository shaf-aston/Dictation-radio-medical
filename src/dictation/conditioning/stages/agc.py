"""AGC: slowly moves the voice toward a target level."""

from __future__ import annotations

import numpy as np

from src.dictation.conditioning.stages.level import PeakFollower, frame_db

# Frames this far under the voice (gaps, hiss) do not steer the gain.
_ADAPT_WITHIN_DB = 10.0


class Agc:
    def __init__(
        self, target_dbfs: float, max_gain_db: float, slew_db_per_sec: float,
        window: int, frames_per_sec: float,
    ) -> None:
        self._target = target_dbfs
        self._max_gain = max_gain_db
        self._step = slew_db_per_sec / frames_per_sec
        self._window = window
        # The voice level falls at the same pace the gain may move.
        self._voice = PeakFollower(slew_db_per_sec, frames_per_sec)
        self._gain_db = 0.0

    def gain(self, power: np.ndarray) -> float:
        db = frame_db(power, self._window)
        voice = self._voice.update(db)
        if db >= voice - _ADAPT_WITHIN_DB:
            want = min(self._max_gain, self._target - voice)
            move = max(-self._step, min(self._step, want - self._gain_db))
            self._gain_db += move
        return 10.0 ** (self._gain_db / 20.0)
