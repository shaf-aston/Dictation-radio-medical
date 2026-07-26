"""Gold-set discovery and manifest I/O for dictation evaluation.

A *gold set* is a directory under ``data/eval/<name>/`` containing audio clips
and a ``manifest.jsonl`` — one JSON record per clip pairing it with its
reference transcript. Four sets exist, and they measure genuinely different
things; reporting them as one blended number would hide exactly the failures
this harness is for:

``own``
    The user's own dictations, their microphone, their accent. The only set
    that measures real acoustics. Irreplaceable — never auto-deleted.
``tts``
    Real radiology report text spoken by an offline synthesiser. Stresses
    medical vocabulary hard under *unrealistically clean* audio, so it scores
    the dictionary, never the microphone. Flagged ``synthetic: true`` and must
    not be quoted as a standalone accuracy figure.
``libri``
    Public-domain read speech. Not medical; the neutral yardstick for "did a
    change break general English" and for per-engine real-time factor.
``bench``
    The two clips already in ``data/bench_audio/``, once hand-corrected.

Audio lives under ``data/`` which is gitignored — deliberate, since ``own`` is
the user's recorded voice and belongs in version control no more than a report
does.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from src.core.json_store import append_jsonl, read_jsonl
from src.features.file_manager import eval_manifest_path, eval_set_dir

logger = logging.getLogger(__name__)

#: A reference carrying this marker is a machine transcription awaiting human
#: correction, not ground truth. :func:`load_set` refuses to score against it —
#: grading a model on its own output produces a flattering number that measures
#: nothing, and is exactly the kind of silent lie this harness exists to catch.
UNREVIEWED_MARKER = "[UNREVIEWED]"

#: Every set the harness knows about, with a one-line description of what it
#: measures. Adding a set means adding it here and to ``build_sets.py``.
KNOWN_SETS: Dict[str, str] = {
    "own": "user's own dictations — real acoustics, real accent",
    "tts": "synthesised radiology reports — medical vocabulary, clean audio",
    "libri": "public-domain read speech — general-English regression + RTF",
    "bench": "existing data/bench_audio clips, hand-corrected",
}


@dataclass(frozen=True)
class Clip:
    """One evaluation clip: an audio file paired with its ground truth."""

    set_name: str
    audio_path: Path
    reference: str
    source: str = ""
    licence: str = ""
    synthetic: bool = False
    duration_sec: float = 0.0

    @property
    def clip_id(self) -> str:
        return f"{self.set_name}/{self.audio_path.stem}"


def write_manifest(set_name: str, clips: List[Clip], replace: bool = True) -> Path:
    """Persist *clips* as ``data/eval/<set_name>/manifest.jsonl``.

    Audio paths are stored relative to the set directory so the corpus survives
    the project being moved or the data directory being copied to another
    machine.
    """
    path = eval_manifest_path(set_name)
    if replace and path.exists():
        path.unlink()

    base = eval_set_dir(set_name)
    records = [
        {
            "audio": _relative_to(c.audio_path, base),
            "reference": c.reference,
            "source": c.source,
            "licence": c.licence,
            "synthetic": c.synthetic,
            "duration_sec": round(c.duration_sec, 3),
        }
        for c in clips
    ]
    append_jsonl(path, records)
    logger.info("Wrote %d clip(s) to %s", len(records), path)
    return path


def load_set(set_name: str) -> List[Clip]:
    """Read one gold set's manifest, skipping records whose audio is missing.

    A missing file is reported loudly rather than silently dropped — a set that
    quietly shrank would make a later run look better than it is.
    """
    path = eval_manifest_path(set_name)
    if not path.exists():
        raise FileNotFoundError(
            f"No manifest for evaluation set '{set_name}' at {path}. "
            f"Build it first: python -m scripts.eval.build_sets --set {set_name}"
        )

    base = eval_set_dir(set_name)
    clips: List[Clip] = []
    missing: List[str] = []
    unreviewed: List[str] = []

    for rec in read_jsonl(path):
        audio = (base / rec["audio"]).resolve()
        if not audio.exists():
            missing.append(rec["audio"])
            continue

        reference = _resolve_reference(audio, rec.get("reference", ""))
        if UNREVIEWED_MARKER in reference:
            unreviewed.append(f"{audio.stem}.reference.txt")
            continue

        clips.append(
            Clip(
                set_name=set_name,
                audio_path=audio,
                reference=reference,
                source=rec.get("source", ""),
                licence=rec.get("licence", ""),
                synthetic=bool(rec.get("synthetic", False)),
                duration_sec=float(rec.get("duration_sec", 0.0)),
            )
        )

    if missing:
        logger.warning(
            "Set '%s': %d manifest entries have no audio file (%s%s) — "
            "scores cover only the %d clips that were found",
            set_name, len(missing), ", ".join(missing[:3]),
            "..." if len(missing) > 3 else "", len(clips),
        )
    if unreviewed:
        raise ValueError(
            f"Evaluation set '{set_name}' has {len(unreviewed)} reference(s) still "
            f"marked {UNREVIEWED_MARKER} and cannot be scored: "
            f"{', '.join(unreviewed[:5])}. Listen to each clip, correct the text in "
            f"{base}, and delete the marker line."
        )
    if not clips:
        raise ValueError(f"Evaluation set '{set_name}' resolved to zero usable clips")
    return clips


def _resolve_reference(audio: Path, manifest_reference: str) -> str:
    """Prefer a hand-editable ``<stem>.reference.txt`` sidecar over the manifest.

    Correcting a draft means editing a text file next to the audio; the sidecar
    therefore wins, so a correction takes effect immediately without anyone
    having to rebuild the manifest or edit JSONL by hand.
    """
    sidecar = audio.parent / f"{audio.stem}.reference.txt"
    if sidecar.exists():
        return sidecar.read_text(encoding="utf-8").strip()
    return manifest_reference


def available_sets() -> List[str]:
    """Known set names that currently have a readable, non-empty manifest."""
    found = []
    for name in KNOWN_SETS:
        try:
            if eval_manifest_path(name).exists() and read_jsonl(eval_manifest_path(name)):
                found.append(name)
        except (OSError, ValueError):
            continue
    return found


def iter_clips(set_names: List[str]) -> Iterator[Clip]:
    """Yield every clip across *set_names* in order."""
    for name in set_names:
        yield from load_set(name)


def audio_duration(path: Path) -> float:
    """Clip length in seconds, or 0.0 if it cannot be read.

    Used for the real-time factor denominator; a zero simply removes that clip
    from the RTF average rather than corrupting it.
    """
    try:
        import soundfile as sf

        return float(sf.info(str(path)).duration)
    except Exception as exc:  # noqa: BLE001 - any decode failure is non-fatal here
        logger.warning("Could not read duration of %s: %s", path, exc)
        return 0.0


def _relative_to(path: Path, base: Path) -> str:
    """Path relative to *base* when possible, else absolute — as a POSIX string."""
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def describe(set_name: str) -> Optional[str]:
    return KNOWN_SETS.get(set_name)
