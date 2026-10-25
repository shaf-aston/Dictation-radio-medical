"""Derive a bad-audio gold set from an existing one, references intact.

The audio conditioning work (docs/audio-conditioning.md) has to prove it helps on
the kinds of bad audio a radiologist meets: a quiet speaker, a noisy room, a
colleague talking nearby. Adding noise, turning down or mixing in another voice
adds no words of the main speaker's, so the source reference stays exactly
correct, the same trick as build_paused_set.py.

Each kind is seeded, so a rebuilt set is the same audio sample for sample.

    quiet   speech turned down by --quiet-db, over a fixed microphone hiss
    noise   steady pink noise (traffic, fans, air) at --snr-db below the speech
    talker  a different report from the same set mixed in --sir-db below the speech

Run:
    python -m scripts.eval.build_degraded_set --from tts --kind quiet
    python -m scripts.eval.replay --set tts_quiet
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import soundfile as sf

from scripts.eval import corpus
from src.dictation.stream.vad import SAMPLE_RATE as SR

KINDS = ("quiet", "noise", "talker")
SEED = 20261023
MIC_HISS_DBFS = -66.0  # a typical USB headset's self-noise, there whether anyone speaks or not


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) or 1e-12


def _db(db: float) -> float:
    return 10.0 ** (db / 20.0)


def _pink(n: int, rng: np.random.Generator) -> np.ndarray:
    """Noise with equal energy per octave: shaped white noise, 1/f in power."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spec), dtype=np.float64)
    f[0] = 1.0
    return np.fft.irfft(spec / np.sqrt(f), n)


def _at_level(x: np.ndarray, rms: float) -> np.ndarray:
    return x * (rms / _rms(x))


def degrade(audio: np.ndarray, kind: str, other: np.ndarray, args, rng: np.random.Generator) -> np.ndarray:
    n = len(audio)
    if kind == "quiet":
        out = audio * _db(-args.quiet_db) + _at_level(rng.standard_normal(n), _db(MIC_HISS_DBFS))
    elif kind == "noise":
        out = audio + _at_level(_pink(n, rng), _rms(audio) * _db(-args.snr_db))
    else:
        other = np.resize(other, n)  # loops a shorter report so the voice never stops
        out = audio + _at_level(other, _rms(audio) * _db(-args.sir_db))
    peak = float(np.max(np.abs(out)))
    return (out / peak * 0.99 if peak > 0.99 else out).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="source", default="tts", help="set to derive from")
    ap.add_argument("--kind", choices=KINDS, required=True)
    ap.add_argument("--name", default="", help="new set name (default <source>_<kind>)")
    ap.add_argument("--quiet-db", type=float, default=30.0, help="quiet: how far the speech is turned down")
    ap.add_argument("--snr-db", type=float, default=5.0, help="noise: speech level over the noise")
    ap.add_argument("--sir-db", type=float, default=6.0, help="talker: main voice over the other voice")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    for flag in ("quiet_db", "snr_db", "sir_db"):
        if not math.isfinite(getattr(args, flag)) or getattr(args, flag) <= 0:
            ap.error(f"--{flag.replace('_', '-')} must be positive")

    name = args.name or f"{args.source}_{args.kind}"
    if corpus.eval_set_dir(name) == corpus.eval_set_dir(args.source):
        # Same folder: each degraded clip would overwrite the clip it was made
        # from, and the own set is the radiologist's voice, which cannot be redone.
        ap.error(f"--name {name!r} is the source set itself; pick another name")
    clips = corpus.load_set(args.source)
    if args.limit:
        clips = clips[: args.limit]
    if args.kind == "talker" and len(clips) < 2:
        ap.error("talker needs at least two clips: the other voice is another clip of the set")

    from faster_whisper.audio import decode_audio

    dest_dir = corpus.eval_set_dir(name)
    dest_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    audios = [decode_audio(str(c.audio_path), sampling_rate=SR) for c in clips]
    detail = {"quiet": f"{args.quiet_db}dB quieter over mic hiss", "noise": f"pink noise at {args.snr_db}dB SNR",
              "talker": f"another report at {args.sir_db}dB below"}[args.kind]

    built = []
    for i, clip in enumerate(clips):
        out = degrade(audios[i], args.kind, audios[(i + 1) % len(clips)], args, rng)
        out_path = dest_dir / clip.audio_path.name
        sf.write(out_path, out, SR)
        built.append(corpus.Clip(
            set_name=name,
            audio_path=out_path,
            reference=clip.reference,      # unchanged: the main speaker said exactly this
            source=f"{clip.source} + {detail}",
            licence=clip.licence,
            synthetic=True,
            duration_sec=len(out) / SR,
        ))

    corpus.write_manifest(name, built)
    print(f"wrote {len(built)} clips to {dest_dir} ({detail})")


if __name__ == "__main__":
    main()
