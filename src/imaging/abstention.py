"""Selective-prediction gate — the scan assistant's central safety mechanism.

Research on clinical CXR models is consistent: high accuracy is not enough;
deployment safety comes from a *rejection mechanism* that abstains on uncertain
cases and defers them to the radiologist. This module is that gate. It takes the
classifier's raw per-label probabilities and keeps only the labels that are

  1. actually trained for the loaded weights (untrained heads emit noise), and
  2. at or above a calibrated per-pathology probability threshold.

Everything else is withheld — the tool says nothing rather than something
uncertain. Thresholds default to a conservative (high) value and are loaded from
``data/imaging/thresholds.json`` when present, else the packaged defaults; a site
can lower a threshold deliberately, but the shipped behaviour errs toward silence.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

from src.imaging.schemas import ImagingFinding, PathologyScore

logger = logging.getLogger(__name__)

_DEFAULT_KEY = "_default"
_FALLBACK_THRESHOLD = 0.85


@lru_cache(maxsize=1)
def _packaged_thresholds() -> Dict[str, float]:
    """Load the packaged default thresholds shipped with the app."""
    path = Path(__file__).parent / "resources" / "thresholds.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read packaged thresholds (%s); using fallback", exc)
        return {_DEFAULT_KEY: _FALLBACK_THRESHOLD}
    return {k: float(v) for k, v in raw.items() if not k.startswith("_comment")}


def load_thresholds() -> Dict[str, float]:
    """Return the active thresholds: user overrides merged over packaged defaults.

    The user copy in ``data/imaging/thresholds.json`` (if any) takes precedence
    per-key, so a site can tune individual pathologies without restating them all.
    """
    thresholds = dict(_packaged_thresholds())
    try:
        from src.features.file_manager import imaging_thresholds_path
        user_path = imaging_thresholds_path()
        if user_path.is_file():
            user = json.loads(user_path.read_text(encoding="utf-8"))
            thresholds |= {
                k: float(v) for k, v in user.items() if not k.startswith("_")
            }
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.warning("Ignoring unreadable user thresholds: %s", exc)
    return thresholds


def threshold_for(label: str, thresholds: Dict[str, float]) -> float:
    """The probability threshold for *label*, falling back to ``_default``."""
    return thresholds.get(label, thresholds.get(_DEFAULT_KEY, _FALLBACK_THRESHOLD))


def apply(scores: List[PathologyScore]) -> List[ImagingFinding]:
    """Filter raw scores into the findings confident enough to consider.

    Returns one :class:`ImagingFinding` per *kept* label (untrained or
    below-threshold labels are dropped entirely), sorted most-confident first.
    ``region`` is left None here — localization fills it, and the analyzer drops
    any finding that still lacks a region before it reaches the user.
    """
    thresholds = load_thresholds()
    kept: List[ImagingFinding] = []
    for s in scores:
        if not s.trained:
            continue  # untrained head → pure noise, never surface
        if s.probability >= threshold_for(s.label, thresholds):
            kept.append(ImagingFinding(
                label=s.label,
                probability=s.probability,
                confident=True,
            ))
    kept.sort(key=lambda f: f.probability, reverse=True)
    return kept
