"""Paths and the one place this experiment reads secrets from .env."""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MODELS = HERE / "models"
KAGGLE_USER = "shafeinhashmi"


def load_env() -> None:
    """Put .env values into os.environ (Kaggle CLI reads KAGGLE_API_TOKEN)."""
    env = HERE / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))
