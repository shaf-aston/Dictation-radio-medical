"""Paths, settings.json, and the one place this experiment reads .env."""
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MODELS = HERE / "models"
SETTINGS = json.loads((HERE / "settings.json").read_text(encoding="utf-8"))
for _k in ("kernel_start_wait_sec", "poll_sec", "max_wait_hours", "smoke_max_wer_gap_pp"):
    # NaN would make every comparison false (a smoke check that always passes); 0 a busy loop.
    if not (isinstance(SETTINGS[_k], (int, float)) and math.isfinite(SETTINGS[_k]) and SETTINGS[_k] > 0):
        raise SystemExit(f"settings.json: {_k} must be a number above 0, got {SETTINGS[_k]!r}")


def load_env() -> None:
    """Put .env values into os.environ (the Kaggle CLI reads KAGGLE_API_TOKEN)."""
    env = HERE / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))
