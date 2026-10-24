"""Chain: cuts audio into overlapping frames, runs the steps, stitches it back.

One FFT per frame. Each step returns a gain, the chain multiplies it in. Frames
are fixed in stream position (hop = half a window), never in block position, so
the output is bit-identical however the audio arrives.

Alignment: output sample n is input sample n. The stream is primed with one hop
of zeros, so the first frame covers [-hop, hop) and nothing is shifted.

Bounds: a non-finite input sample becomes 0 (or full scale), so one bad sample
cannot poison a step's state for the rest of the dictation; output is clipped
to [-1, 1], the range every engine expects.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from src.core import perf
from src.dictation.conditioning.port import Step


# The largest window accepted: one window is held back, and past this the
# delay before words appear starts to show.
MAX_WINDOW = 4096


class Chain:
    def __init__(self, steps: Sequence[Step], window: int) -> None:
        if (not isinstance(window, int) or isinstance(window, bool)
                or not 0 < window <= MAX_WINDOW or window % 2):
            raise ValueError(f"window must be an even integer in (0, {MAX_WINDOW}], got {window!r}")
        self._steps = list(steps)
        self._window = window
        self._hop = window // 2
        # Periodic Hann, square-rooted: applied going in AND out, so the pair
        # sums to 1 at 50% overlap and unity gains give the input back.
        n = np.arange(window)
        self._win = np.sqrt(0.5 - 0.5 * np.cos(2.0 * np.pi * n / window))
        self._in = np.zeros(self._hop, dtype=np.float32)   # the priming zeros
        self._acc = np.zeros(window, dtype=np.float64)     # overlap-add in progress
        self._skip = self._hop    # priming samples still to drop from the output
        self._total_in = 0
        self._returned = 0
        self._spent = False

    @property
    def spent(self) -> bool:
        """True after flush, or after a step failed part-way through a frame
        (the state has moved on without its output, so nothing after lines up)."""
        return self._spent

    def process(self, block: np.ndarray) -> np.ndarray:
        """Float32 mono in. Returns the cleaned samples that are ready."""
        if self._spent:
            raise RuntimeError("chain was flushed or failed; build a new one")
        block = np.nan_to_num(np.asarray(block, dtype=np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
        if not self._steps:
            return block
        with perf.stage("audio.condition"):
            self._total_in += len(block)
            out = self._run(block)
            self._returned += len(out)
            return out

    def flush(self) -> np.ndarray:
        """Stop: returns everything still held. After it, total out == total in."""
        if self._spent:
            raise RuntimeError("chain was flushed or failed; build a new one")
        self._spent = True
        if not self._steps:
            return np.zeros(0, dtype=np.float32)
        parts: List[np.ndarray] = []
        got = self._returned
        while got < self._total_in:
            part = self._run(np.zeros(self._hop, dtype=np.float32))
            parts.append(part)
            got += len(part)
        out = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        return out[: self._total_in - self._returned]

    def _run(self, block: np.ndarray) -> np.ndarray:
        """Take every frame the input now allows; return the finished samples."""
        w, h = self._window, self._hop
        self._in = np.concatenate([self._in, block])
        done: List[np.ndarray] = []
        while len(self._in) >= w:
            try:
                done.append(self._frame(self._in[:w]))
            except Exception:
                self._spent = True
                raise
            self._in = self._in[h:]
        if not done:
            return np.zeros(0, dtype=np.float32)
        out = np.concatenate(done)
        drop = min(self._skip, len(out))
        self._skip -= drop
        return out[drop:]

    def _frame(self, frame: np.ndarray) -> np.ndarray:
        """One frame through the steps; returns its finished first hop."""
        h = self._hop
        spec = np.fft.rfft(frame * self._win)
        power = spec.real ** 2 + spec.imag ** 2
        for step in self._steps:
            spec = spec * step.gain(power)
            power = spec.real ** 2 + spec.imag ** 2
        self._acc += np.fft.irfft(spec, self._window) * self._win
        ready = np.clip(self._acc[:h], -1.0, 1.0).astype(np.float32)
        self._acc = np.concatenate([self._acc[h:], np.zeros(h)])
        return ready
