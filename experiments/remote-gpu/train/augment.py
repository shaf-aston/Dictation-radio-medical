"""Runs ON Kaggle (shipped with the training data): noisy copies of training clips.

Babble (another clip), recorded noise, or steady hiss, at a random loudness gap; some also in a real
room's echo. Not pink noise: the local noise eval uses pink, so training on it would flatter it.
"""
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

from sources import SR, decode


def real_noises_and_rooms(url: str, tmp: Path) -> tuple:
    """OpenSLR 28 (Apache 2.0): recorded noises, and real room echoes (impulse responses); channels are averaged by decode()."""
    z = tmp / "rirs_noises.zip"
    subprocess.run(["curl", "-sSfL", "-o", str(z), url], check=True)
    subprocess.run(["unzip", "-q", "-o", str(z), "-d", str(tmp)], check=True)
    files = sorted((tmp / "RIRS_NOISES").rglob("*.wav"))
    noises = [f for f in files if "pointsource_noises" in str(f) or ("real_rirs" in str(f) and "noise" in f.name)]
    rooms = [f for f in files if "real_rirs" in str(f) and "rir" in f.name and "noise" not in f.name]
    return noises, rooms


def _rms(x) -> float:
    return float(np.sqrt(np.mean(x ** 2))) + 1e-9


def add_noisy_copies(rows: list, cfg: dict, rng, tmp: Path, clips: Path) -> dict:
    """Appends a noisy copy of cfg["noisy_copy_fraction"] of `rows` to `rows`; returns counts for the log."""
    np_rng = np.random.default_rng(cfg["seed"])
    lo, hi = cfg["noise_snr_db"]
    noises, rooms = real_noises_and_rooms(cfg["rir_noise_url"], tmp)
    picks = rng.sample(rows, int(len(rows) * cfg["noisy_copy_fraction"]))
    for i, r in enumerate(picks):
        a = sf.read(r["audio_filepath"], dtype="float32")[0]
        if rng.random() < cfg["reverb_fraction"]:
            room = decode(rng.choice(rooms).read_bytes())
            a = fftconvolve(a, room / (np.max(np.abs(room)) + 1e-9))[: len(a)].astype(np.float32)
        kind = rng.random()
        if kind < 1 / 3:
            noise = np.resize(sf.read(rng.choice(rows)["audio_filepath"], dtype="float32")[0], len(a))
        elif kind < 2 / 3:
            noise = np.resize(decode(rng.choice(noises).read_bytes()), len(a))
        else:
            noise = np_rng.standard_normal(len(a)).astype(np.float32)
        gap = 10 ** (rng.uniform(lo, hi) / 20)
        mixed = np.clip(a + noise * (_rms(a) / _rms(noise) / gap), -1, 1)
        path = clips / f"noisy_{i:05d}.flac"
        sf.write(path, mixed, SR)
        rows.append({**r, "audio_filepath": str(path), "source": r["source"] + "+noise"})
    return {"openslr28": {"noises": len(noises), "rooms": len(rooms)}, "noisy_copies": len(picks)}
