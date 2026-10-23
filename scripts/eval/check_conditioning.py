"""Check the audio-conditioning skeleton: alignment, block-split identity, length.

Run:
    python -m scripts.eval.check_conditioning

Exits non-zero on the first failed check. Uses steps defined here, so it needs
no real cleaning step and stays valid as steps are added.
"""

from __future__ import annotations

import sys

import numpy as np

from src.core import perf
from src.dictation.conditioning.chain import Chain
from src.dictation.conditioning.factory import build_chain

WINDOW = 512
BLOCK = 320                 # 20 ms at 16 kHz, the browser's block
P95_LIMIT_MS = 0.5          # docs/audio-conditioning.md pass mark


class Unity:
    def gain(self, power: np.ndarray) -> float:
        return 1.0


class Drift:
    """Stateful: the gain changes every frame, so a frame run in the wrong
    order or the wrong number of times shows up in the output."""

    def __init__(self) -> None:
        self.n = 0

    def gain(self, power: np.ndarray) -> np.ndarray:
        self.n += 1
        return np.full(len(power), 0.5 + 0.4 * np.sin(self.n * 0.7))


class Tilt:
    """Trivial per-bin gain, for the timing run."""

    def gain(self, power: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + 0.001 * np.arange(len(power)))


def run(steps, audio: np.ndarray, block: int) -> np.ndarray:
    chain = Chain(steps, WINDOW)
    parts = [chain.process(audio[i : i + block]) for i in range(0, len(audio), block)]
    parts.append(chain.flush())
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def check(ok: bool, what: str) -> None:
    print(("ok    " if ok else "FAIL  ") + what)
    if not ok:
        sys.exit(1)


def raises_value_error(cfg: dict) -> bool:
    try:
        build_chain(cfg, 16000)
    except ValueError:
        return True
    return False


def main() -> None:
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(16000) * 0.1).astype(np.float32)

    out = run([Unity()], audio, 777)
    err = np.abs(out.astype(np.float64) - audio)
    check(len(out) == len(audio) and err.max() < 1e-6,
          f"unity chain gives the input back (max error {err.max():.2e})")
    check(err[:512].max() < 1e-6 and err[-512:].max() < 1e-6,
          "first and last 512 samples match")

    ref = run([Drift()], audio, len(audio))
    for block in (1, 160, 320, 511, 512, 4000):
        check(np.array_equal(run([Drift()], audio, block), ref),
              f"stateful chain identical at block size {block}")

    for n in (0, 1, 255, 256, 257, 16000):
        check(len(run([Unity()], audio[:n], BLOCK)) == n, f"length in == out for {n} samples")

    empty = Chain([], WINDOW)
    check(np.array_equal(empty.process(audio[:300]), audio[:300])
          and len(empty.flush()) == 0, "empty chain returns the input unchanged")

    for cfg in ({"window_samples": 0}, {"window_samples": -2},
                {"window_samples": 511}, {"nonsense": {"on": True}}):
        check(raises_value_error(cfg), f"factory rejects {cfg}")

    perf.reset()
    chain = Chain([Tilt(), Tilt(), Tilt(), Tilt()], WINDOW)
    long_audio = (rng.standard_normal(16000 * 30) * 0.1).astype(np.float32)
    for i in range(0, len(long_audio), BLOCK):
        chain.process(long_audio[i : i + BLOCK])
    p95 = perf.snapshot()["audio.condition"]["p95_ms"]
    print(f"audio.condition p95 per {BLOCK}-sample block, 4 steps: {p95:.3f} ms")
    check(p95 <= P95_LIMIT_MS, f"p95 under {P95_LIMIT_MS} ms")


if __name__ == "__main__":
    main()
