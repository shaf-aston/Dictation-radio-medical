"""Raw text per clip for one model on one eval set, kept so re-scoring (a new correction rule, a new
feature) takes seconds: {audio file name: [text, decode seconds or None]}; None = decoded on Kaggle's
GPU, so no CPU timing. compare.py reads and fills it; run_train.py files the kernel's decodes into it.
A version is never edited after it is saved, so its decodes never go stale. Lives in the app's
rebuildable cache folder (data/cache/compare_decodes/<set>/<model>.json)."""
import json

import config  # noqa: F401  (puts the repo root on sys.path)
from src.core.paths import cache_dir


def _path(set_name: str, model: str):
    return cache_dir() / "compare_decodes" / set_name / f"{model}.json"


def load(set_name: str, model: str) -> dict:
    p = _path(set_name, model)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save(set_name: str, model: str, decodes: dict) -> None:
    p = _path(set_name, model)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(decodes), encoding="utf-8")
