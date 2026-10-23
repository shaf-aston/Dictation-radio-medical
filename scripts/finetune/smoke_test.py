"""Run the Kaggle notebook's code on CPU: the code path, not the result.

    python -m scripts.finetune.build_dataset --voices <voices> --out /tmp/rad-smoke --limit 12 --copies 1
    <python with nemo_toolkit[asr], jiwer, onnx-asr> scripts/finetune/smoke_test.py \\
        --data /tmp/rad-smoke --out /tmp/rad-smoke-out

Executes every code cell of ``parakeet_kaggle.ipynb`` in order (shell lines
dropped), with ``SMOKE = True``: data loading, model, optimiser and the
before/after measurement, then the real export, then loading the export with
onnx-asr exactly as the app does. A cell that fails here would fail on Kaggle
too, before any GPU time is spent. The one step it cannot run is
``trainer.fit``: NeMo's TDT loss is CUDA-only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.finetune.make_notebook import code_cells  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cells = code_cells()
    scope: dict = {}
    exec(cells[0], scope)  # settings
    scope.update(DATA_DIR=str(args.data), OUT_DIR=str(args.out), SMOKE=True)
    for i, cell in enumerate(cells[1:], start=2):
        print(f"--- cell {i} ---", flush=True)
        exec(cell, scope)
    print("SMOKE TEST PASSED: data -> model -> optimiser -> export -> onnx-asr load "
          "(the training step itself needs a GPU: NeMo's TDT loss is CUDA-only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
