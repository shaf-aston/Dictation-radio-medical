"""End-to-end latency benchmark for the dictation post-process pipeline.

Runs the *real* pipeline (no mocks) and reports where the time goes, using the
same :mod:`src.core.perf` timings the live app records. Three uses:

1. **Terminal latency check** (default) — warm-up cost, first-chunk (cold) cost,
   and steady-state (warm) per-chunk cost, plus the per-stage perf table::

       python scripts/benchmark.py

2. **Manual / visual check** — feed your own dictation text and see exactly what
   the pipeline turns it into (does it help or hurt your words?)::

       python scripts/benchmark.py --text "there is a tier of the supraspinatus"
       python scripts/benchmark.py --file my_dictation.txt --accent south_asian

3. **Regression guard** — fail (exit 1) if the warm per-chunk latency drifts
   above a budget, so a future change that slows the hot path is caught::

       python scripts/benchmark.py --max-warm-ms 60

Whisper model load is *not* exercised here (it needs audio + the model files);
this benchmark isolates the text pipeline, which is the part that runs on every
live cycle. Use ``--warm`` to include the background warm-up in the timing.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow running as `python scripts/benchmark.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core import perf  # noqa: E402
from src.dictation.postprocess import postprocess_transcript  # noqa: E402
from src.dictation import warmup  # noqa: E402

# A representative slice of dictated radiology, with the kinds of errors the
# pipeline exists to fix (mis-transcriptions, spoken punctuation, measurements).
_SAMPLE = (
    "the lungs are clear with no focal consolidation the heart size is normal "
    "there is a tier of the supraspinatus tendon the femer shows no fracture "
    "5 by 3 by 2 millimetres lesion findings colon no acute abnormality end period"
)


def _time_ms(fn) -> float:
    start = time.perf_counter()
    fn()
    return (time.perf_counter() - start) * 1000.0


def _run_manual(text: str, accent: str, cleanup_level: str) -> None:
    """Show the before/after for a single supplied transcript."""
    warmup.warm_up()  # so the first (only) run is not paying build cost
    out = postprocess_transcript(text, accent=accent, cleanup_level=cleanup_level)
    print("\n--- input ----------------------------------------------------")
    print(text)
    print("\n--- output ---------------------------------------------------")
    print(out)
    print()


def _run_benchmark(
    accent: str, cleanup_level: str, iterations: int, include_warm: bool
) -> float:
    """Measure warm-up, cold-first-chunk, and warm per-chunk latency.

    Returns the mean warm per-chunk latency in ms (used for the budget guard).
    """
    if include_warm:
        warm_ms = _time_ms(warmup.warm_up)
        print(f"warm-up (spelling index build/load) : {warm_ms:8.1f} ms")

    def _one() -> None:
        postprocess_transcript(_SAMPLE, accent=accent, cleanup_level=cleanup_level)

    cold_ms = _time_ms(_one)
    print(f"first chunk (cold, if not warmed)   : {cold_ms:8.1f} ms")

    warm_samples = [_time_ms(_one) for _ in range(iterations)]
    mean_warm = sum(warm_samples) / len(warm_samples)
    peak_warm = max(warm_samples)
    print(
        f"warm chunk  (n={iterations:<3d} mean/peak)      : "
        f"{mean_warm:8.1f} / {peak_warm:.1f} ms"
    )

    print("\n--- per-stage timings (slowest total first) ------------------")
    stats = perf.snapshot()
    print(f"{'stage':40s} {'n':>4s} {'mean':>9s} {'p95':>9s} {'max':>9s}")
    for name, s in stats.items():
        print(
            f"{name:40s} {int(s['count']):>4d} "
            f"{s['mean_ms']:>7.1f}ms {s['p95_ms']:>7.1f}ms {s['max_ms']:>7.1f}ms"
        )
    return mean_warm


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", help="Run the pipeline on this text and print the result.")
    parser.add_argument("--file", help="Run the pipeline on the contents of this file.")
    parser.add_argument("--accent", default="neutral", help="Accent profile (default: neutral).")
    parser.add_argument(
        "--cleanup-level", default="medium", choices=("soft", "medium", "hard"),
        help="Pipeline cleanup level (default: medium).",
    )
    parser.add_argument("--iterations", type=int, default=20, help="Warm samples to average.")
    parser.add_argument(
        "--warm", action="store_true",
        help="Include the background warm-up (spelling index) in the timing.",
    )
    parser.add_argument(
        "--max-warm-ms", type=float, default=None,
        help="Fail (exit 1) if mean warm per-chunk latency exceeds this budget.",
    )
    parser.add_argument("--verbose", action="store_true", help="Show INFO logs.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    perf.reset()

    manual_text = args.text
    if args.file:
        manual_text = Path(args.file).read_text(encoding="utf-8")

    if manual_text is not None:
        _run_manual(manual_text, args.accent, args.cleanup_level)
        return 0

    mean_warm = _run_benchmark(
        args.accent, args.cleanup_level, args.iterations, include_warm=args.warm,
    )

    if args.max_warm_ms is not None and mean_warm > args.max_warm_ms:
        print(
            f"\nFAIL: warm latency {mean_warm:.1f} ms exceeds budget "
            f"{args.max_warm_ms:.1f} ms",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
