"""The one thing a cleaning step must do: look at a frame, return a gain."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Step(Protocol):
    def gain(self, power: np.ndarray) -> np.ndarray | float:
        """power: |X|^2 of this frame after earlier steps, float64, window//2+1 bins.
        Returns an amplitude gain per bin (array) or for the whole frame (float).
        May keep state from frame to frame; never sees block boundaries."""
        ...
