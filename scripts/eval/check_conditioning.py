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
from src.dictation.conditioning.factory import ORDER, build_chain

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


ALL_ON = {name: {"on": True} for name in ORDER}


def real_chain() -> Chain:
    """Every shipped step switched on, built the way the app builds it."""
    return build_chain(ALL_ON, 16000)


def run(steps, audio: np.ndarray, block: int) -> np.ndarray:
    chain = steps() if callable(steps) else Chain(steps, WINDOW)
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

    # Speech-like input (bursts and gaps) so every real step has work to do.
    t = np.arange(16000 * 3) / 16000
    burst = (np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 1.5 * t) > 0)).astype(np.float32)
    voiced = (burst * 0.3 + rng.standard_normal(len(t)).astype(np.float32) * 0.01)
    real_ref = run(real_chain, voiced, len(voiced))
    check(len(real_ref) == len(voiced) and bool(np.isfinite(real_ref).all()),
          "all four real steps: length kept, output finite")
    for block in (1, 160, 320, 511, 512, 4000):
        check(np.array_equal(run(real_chain, voiced, block), real_ref),
              f"all four real steps identical at block size {block}")
    for n in (0, 1, 255, 256, 257, 16000):
        check(len(run(real_chain, voiced[:n], BLOCK)) == n,
              f"all four real steps: length in == out for {n} samples")

    for n in (0, 1, 255, 256, 257, 16000):
        check(len(run([Unity()], audio[:n], BLOCK)) == n, f"length in == out for {n} samples")

    empty = Chain([], WINDOW)
    check(np.array_equal(empty.process(audio[:300]), audio[:300])
          and len(empty.flush()) == 0, "empty chain returns the input unchanged")

    for cfg in ({"window_samples": 0}, {"window_samples": -2},
                {"window_samples": 511}, {"window_samples": 8192},
                {"window_samples": True}, {"nonsense": {"on": True}},
                {"agc": None}, {"agc": {"on": "yes"}}, {"agc": {"gain": 3}},
                {"agc": {"on": True, "target_dbfs": 5.0}}, {"agc": {"max_gain_db": float("inf")}},
                {"denoise": {"on": True, "min_gain_db": 3.0}}, {"denoise": {"on": True, "smoothing": 1.0}},
                {"highpass": {"on": True, "cutoff_hz": 9000.0}}):
        # Ranges are checked on a step that is on; types and names on every step.
        check(raises_value_error(cfg), f"factory rejects {cfg}")

    bad = voiced.copy()
    bad[1000:1010] = np.nan
    bad[2000] = np.inf
    out = run(real_chain, bad, BLOCK)
    check(len(out) == len(bad) and bool(np.isfinite(out).all())
          and float(np.abs(out).max()) <= 1.0,
          "NaN/inf input: length kept, output finite and within full scale")

    spent = real_chain()
    spent.flush()
    try:
        spent.process(audio[:BLOCK])
        reused = True
    except RuntimeError:
        reused = False
    check(spent.spent and not reused, "a flushed chain refuses more audio")

    perf.reset()
    chain = real_chain()
    long_audio = (rng.standard_normal(16000 * 30) * 0.1).astype(np.float32)
    for i in range(0, len(long_audio), BLOCK):
        chain.process(long_audio[i : i + BLOCK])
    p95 = perf.snapshot()["audio.condition"]["p95_ms"]
    print(f"audio.condition p95 per {BLOCK}-sample block, all 4 real steps: {p95:.3f} ms")
    check(p95 <= P95_LIMIT_MS, f"p95 under {P95_LIMIT_MS} ms")


if __name__ == "__main__":
    main()
