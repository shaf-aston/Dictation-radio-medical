"""Highpass: rumble and handling thump out. Stateless."""

from __future__ import annotations

import numpy as np


class Highpass:
    def __init__(self, cutoff_hz: float, sr: int, window: int) -> None:
        freq = np.fft.rfftfreq(window, 1.0 / sr)
        # Nothing below half the cutoff, a smooth rise to full gain at it.
        t = np.clip((freq - cutoff_hz / 2.0) / (cutoff_hz / 2.0), 0.0, 1.0)
        self._gain = 0.5 - 0.5 * np.cos(np.pi * t)

    def gain(self, power: np.ndarray) -> np.ndarray:
        return self._gain
