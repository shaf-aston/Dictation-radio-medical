"""Runs ON Kaggle (shipped with the training data): reading audio from wherever it comes from,
as 16 kHz mono float32 with its text."""
import io
import os
import subprocess
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfApi, hf_hub_download
from scipy.signal import resample_poly

from text_rules import normalise

SR = 16000


def decode(b: bytes) -> np.ndarray:
    """Any encoded clip (wav/flac/mp3) -> 16 kHz mono float32."""
    try:
        a, sr = sf.read(io.BytesIO(b), dtype="float32")
    except Exception:
        r = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", str(SR),
                            "pipe:1"], input=b, capture_output=True, check=True)
        return np.frombuffer(r.stdout, np.float32)
    if a.ndim > 1:
        a = a.mean(axis=1)
    return a if sr == SR else resample_poly(a, SR, sr).astype(np.float32)


def read_clip(path: str) -> np.ndarray:
    return decode(Path(path).read_bytes())


def parallel(fn, items, chunk: int = 16):
    """fn over items on every CPU core, results in order."""
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        yield from ex.map(fn, items, chunksize=chunk)


def column(names: list, *wanted: str) -> str:
    for w in wanted:
        if w in names:
            return w
    raise SystemExit(f"none of {wanted} in columns {names}")


def hf_files(repo: str, prefix: str, n: int, cache: Path) -> list:
    """`n` parquet shards spread evenly across the repo's `prefix` shards (EdAcc groups shards by speaker)."""
    names = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                   if f.startswith(prefix) and f.endswith(".parquet"))
    if not names:
        raise SystemExit(f"no {prefix}*.parquet in {repo}")
    pick = names[:: max(1, len(names) // n)][:n]
    return [hf_hub_download(repo, f, repo_type="dataset", local_dir=cache) for f in pick]


def parquet_rows(files: list, text_cols: tuple, rng, audio_col: str = "audio"):
    """Yields (audio bytes, text) from parquet shards, shuffled within each shard."""
    for f in files:
        pf = pq.ParquetFile(f)
        tcol = column(pf.schema_arrow.names, *text_cols)
        rows = pf.read(columns=[tcol, audio_col]).to_pylist()
        rng.shuffle(rows)
        for r in rows:
            a = r[audio_col]
            yield (a["bytes"] if isinstance(a, dict) else a), r[tcol]


def eval_set(rows, min_sec: float, max_sec: float, n: int) -> list:
    """Up to `n` (audio, text) pairs with text and a length in [min_sec, max_sec]; audio as bytes or already decoded."""
    out = []
    for b, text in rows:
        a = decode(b) if isinstance(b, bytes) else b
        if normalise(text) and min_sec <= len(a) / SR <= max_sec:
            out.append((a, text))
        if len(out) >= n:
            break
    return out
