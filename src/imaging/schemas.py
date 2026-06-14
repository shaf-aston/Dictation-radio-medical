"""Data contracts for the scan-assistant subsystem.

Plain dataclasses (no ML types) so they can be imported, serialised, and tested
without torch installed — matching the project's lightweight, file-first style.
"""


from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Standard disclaimer attached to every result. Outputs are assistive only.
DISCLAIMER = (
    "AI-assisted suggestion for radiologist review only — NOT a diagnosis. "
    "This model is not FDA-cleared or CE-marked. Confirm all findings clinically."
)

# A region is an axis-aligned box in *fractional* image coordinates (0–1), so it
# survives any later display resize: (x0, y0, x1, y1).
Region = Tuple[float, float, float, float]


@dataclass
class PathologyScore:
    """Raw per-label model output before the abstention gate is applied."""

    label: str
    probability: float
    trained: bool   # False if the loaded model has no real head for this label


@dataclass
class ImagingFinding:
    """A finding the model is confident enough to surface, with its evidence.

    A finding only ever reaches the UI if ``confident`` is True AND ``region`` is
    present; the analyzer drops anything that cannot satisfy both (see
    :mod:`src.imaging.analyzer`).
    """

    label: str
    probability: float
    confident: bool
    region: Optional[Region] = None   # Grad-CAM bounding box, fractional coords


@dataclass
class ImagingResult:
    """The full outcome of analysing one image."""

    findings: List[ImagingFinding] = field(default_factory=list)
    overlay_png_path: Optional[str] = None   # heatmap overlay for the kept findings
    all_scores: List[PathologyScore] = field(default_factory=list)  # for debugging/audit
    disclaimer: str = DISCLAIMER
    embedding: Optional[object] = None   # normalized (L2) embedding for retrieval; cached for reuse

    @property
    def has_findings(self) -> bool:
        return bool(self.findings)
