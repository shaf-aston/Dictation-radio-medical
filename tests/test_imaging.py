"""Tests for the scan-assistant safety logic.

These guard the patient-safety invariants of ``src.imaging`` WITHOUT requiring
torch or downloading weights — a fake classifier supplies probabilities, and we
assert the abstention gate and analyzer behave conservatively:

  * untrained heads are never surfaced (they emit noise);
  * below-threshold findings are withheld;
  * a confident finding with no localised region is withheld;
  * every result carries the non-diagnostic disclaimer.
"""

from __future__ import annotations

from typing import List

from src.imaging import abstention
from src.imaging.analyzer import analyze_scan
from src.imaging.schemas import DISCLAIMER, PathologyScore


# ---------------------------------------------------------------------------
# Fakes — stand in for the torch-backed classifier / localization
# ---------------------------------------------------------------------------

class _FakeClassifier:
    """Returns canned scores; satisfies the Classifier protocol for analyze_scan."""

    def __init__(self, scores: List[PathologyScore]) -> None:
        self._scores = scores

    def predict(self, image_path: str) -> List[PathologyScore]:
        return self._scores


def _patch_localization(monkeypatch, regions: dict) -> None:
    """Make localization deterministic: return *regions*, render no overlay."""
    from src.imaging import analyzer
    monkeypatch.setattr(analyzer.localization, "localize",
                        lambda clf, path, findings: dict(regions))
    monkeypatch.setattr(analyzer.localization, "render_overlay",
                        lambda path, findings, out: "overlay.png")


# ---------------------------------------------------------------------------
# Abstention gate
# ---------------------------------------------------------------------------

def test_untrained_head_never_surfaced():
    scores = [
        PathologyScore(label="Pneumonia", probability=0.99, trained=False),
    ]
    assert abstention.apply(scores) == []


def test_below_threshold_withheld():
    # 0.50 is well under the conservative default (~0.85+).
    scores = [PathologyScore(label="Effusion", probability=0.50, trained=True)]
    assert abstention.apply(scores) == []


def test_above_threshold_kept_and_sorted():
    scores = [
        PathologyScore(label="Effusion", probability=0.86, trained=True),
        PathologyScore(label="Cardiomegaly", probability=0.97, trained=True),
    ]
    kept = abstention.apply(scores)
    assert [f.label for f in kept] == ["Cardiomegaly", "Effusion"]
    assert all(f.confident for f in kept)


def test_user_thresholds_override_defaults():
    import json

    from src.features.file_manager import imaging_thresholds_path  # type: ignore
    # Lower one pathology's bar so a moderate probability now passes.
    imaging_thresholds_path().write_text(
        json.dumps({"Effusion": 0.4}), encoding="utf-8")
    scores = [PathologyScore(label="Effusion", probability=0.5, trained=True)]
    assert [f.label for f in abstention.apply(scores)] == ["Effusion"]


# ---------------------------------------------------------------------------
# Analyzer invariants
# ---------------------------------------------------------------------------

def test_confident_without_region_is_withheld(monkeypatch):
    clf = _FakeClassifier([
        PathologyScore(label="Cardiomegaly", probability=0.97, trained=True),
    ])
    # Localization returns no region for the finding → it must be withheld.
    _patch_localization(monkeypatch, regions={})
    result = analyze_scan("dummy.png", classifier=clf)
    assert result.findings == []
    assert result.disclaimer == DISCLAIMER


def test_confident_with_region_is_returned(monkeypatch):
    clf = _FakeClassifier([
        PathologyScore(label="Cardiomegaly", probability=0.97, trained=True),
    ])
    _patch_localization(monkeypatch, regions={"Cardiomegaly": (0.1, 0.2, 0.5, 0.6)})
    result = analyze_scan("dummy.png", classifier=clf)
    assert [f.label for f in result.findings] == ["Cardiomegaly"]
    assert result.findings[0].region == (0.1, 0.2, 0.5, 0.6)
    assert result.overlay_png_path == "overlay.png"


def test_no_candidates_returns_empty_with_disclaimer(monkeypatch):
    clf = _FakeClassifier([
        PathologyScore(label="Effusion", probability=0.10, trained=True),
    ])
    result = analyze_scan("dummy.png", classifier=clf)
    assert not result.has_findings
    assert result.disclaimer == DISCLAIMER
    # Raw scores are retained for audit even when nothing is surfaced.
    assert len(result.all_scores) == 1


def test_heatmap_to_region_math():
    import numpy as np
    from src.imaging.localization import _heatmap_to_region
    cam = np.zeros((10, 10), dtype=float)
    cam[2:5, 3:7] = 1.0   # hot block
    region = _heatmap_to_region(cam, np)
    assert region is not None
    x0, y0, x1, y1 = region
    assert 0.0 <= x0 < x1 <= 1.0
    assert 0.0 <= y0 < y1 <= 1.0
    # The hot block sits in the upper-left quadrant.
    assert x0 == 0.3 and y0 == 0.2
