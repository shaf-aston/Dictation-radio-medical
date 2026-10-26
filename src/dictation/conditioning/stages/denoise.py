"""Denoise: spectral subtraction against a per-bin noise floor it tracks itself."""

from __future__ import annotations

import numpy as np

# Per-bin power the noise floor never drops under (far below any real hiss).
_FLOOR_MIN = 1e-10


class Denoise:
    def __init__(
        self, strength: float, min_gain_db: float, noise_rise_db_per_sec: float,
        smoothing: float, frames_per_sec: float,
    ) -> None:
        self._strength = strength
        self._min_gain = 10.0 ** (min_gain_db / 20.0)
        self._rise = 10.0 ** (noise_rise_db_per_sec / frames_per_sec / 10.0)
        self._smoothing = smoothing
        self._smooth: np.ndarray | None = None
        self._floor: np.ndarray | None = None

    def gain(self, power: np.ndarray) -> np.ndarray | float:
        if self._floor is None:
            if not power.any():
                return 1.0   # digital silence says nothing about the room yet
            # Start from the first frames, not from zero: until speech starts
            # they are the room.
            self._floor = power.copy()
        if self._smooth is None:
            self._smooth = power.copy()
        # A few frames averaged, or the floor would sit on the deepest dips of
        # a random noise spectrum and remove almost nothing.
        self._smooth = self._smoothing * self._smooth + (1.0 - self._smoothing) * power
        # Falls at once, rises slowly. Held above _FLOOR_MIN: a floor at 0 (a
        # muted stretch) would multiply up to 0 forever and switch denoise off.
        self._floor = np.minimum(self._smooth, np.maximum(self._floor, _FLOOR_MIN) * self._rise)
        ratio = self._strength * self._floor / np.maximum(power, 1e-12)
        # Never fully off: a gain floor stops the "musical noise" of bins
        # flickering between on and off.
        return np.maximum(np.sqrt(np.maximum(1.0 - ratio, 0.0)), self._min_gain)
