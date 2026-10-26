"""AGC: slowly moves the voice toward a target level."""

from __future__ import annotations

import numpy as np

from src.dictation.conditioning.stages.level import PeakFollower, frame_db


class Agc:
    def __init__(
        self, target_dbfs: float, max_gain_db: float, slew_db_per_sec: float,
        voice_release_db_per_sec: float, adapt_within_db: float,
        window: int, frames_per_sec: float,
    ) -> None:
        self._target = target_dbfs
        self._max_gain = max_gain_db
        self._step = slew_db_per_sec / frames_per_sec
        # Frames further than this under the voice (gaps, hiss) do not steer the gain.
        self._within = adapt_within_db
        self._window = window
        # The voice level falls slowly in a pause, so the hiss between words
        # stays well under it and is never turned up.
        self._voice = PeakFollower(voice_release_db_per_sec, frames_per_sec)
        self._gain_db = 0.0

    def gain(self, power: np.ndarray) -> float:
        db = frame_db(power, self._window)
        voice = self._voice.update(db)
        if db >= voice - self._within:
            want = min(self._max_gain, self._target - voice)
            # Down at once, up slowly: the first word after a pause never
            # arrives at a gain set for the quiet before it.
            self._gain_db = want if want < self._gain_db else min(want, self._gain_db + self._step)
        return 10.0 ** (self._gain_db / 20.0)
