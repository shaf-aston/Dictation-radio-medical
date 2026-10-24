"""Paths, settings.json, and the one place this experiment reads .env."""
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MODELS = HERE / "models"
SETTINGS = json.loads((HERE / "settings.json").read_text(encoding="utf-8"))


def load_env() -> None:
    """Put .env values into os.environ (the Kaggle CLI reads KAGGLE_API_TOKEN)."""
    env = HERE / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))
