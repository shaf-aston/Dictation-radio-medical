"""Data contracts for the scan-assistant subsystem.

Plain dataclasses (no ML types) so they can be imported, serialised, and tested
without torch installed — matching the project's lightweight, file-first style.
"""


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Standard disclaimer attached to every result. Outputs are assistive only.
DISCLAIMER = (
    "AI-assisted suggestion for radiologist review only — NOT a diagnosis. "
    "This model is not FDA-cleared or CE-marked. Confirm all findings clinically."
)

# Default TorchXRayVision base weights (every output head trained, "all"). Single
# source of truth shared by the classifier and the scan fine-tuning task.
DEFAULT_SCAN_WEIGHTS = "densenet121-res224-all"

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


@dataclass
class ReferenceCase:
    """One labelled reference X-ray in the retrieval index.

    ``labels`` are the per-pathology binary flags (``{"Fracture": 1, ...}``) used
    for diagnosis filtering; ``attributes`` carry optional clinical context
    (``age``, ``sex``, ``bmi``, ``view``, ``history``) used to narrow matches.
    Both default empty so a legacy sidecar (labels only) still produces a valid
    case — attribute filters simply don't apply to it.
    """

    path: str
    labels: Dict[str, int] = field(default_factory=dict)
    attributes: Dict[str, object] = field(default_factory=dict)

    def to_metadata(self) -> Dict[str, object]:
        """Serialise to the plain dict persisted in the index metadata JSON."""
        return {"path": self.path, "labels": self.labels, "attributes": self.attributes}

    @classmethod
    def from_metadata(cls, meta: object) -> "ReferenceCase":
        """Rebuild from persisted metadata, tolerating the legacy bare-path form.

        Older indices stored each entry as just the image path string; wrap those
        as an attribute-less case so old caches keep loading.
        """
        if isinstance(meta, str):
            return cls(path=meta)
        if isinstance(meta, dict):
            return cls(
                path=str(meta.get("path", "")),
                labels=dict(meta.get("labels", {})),
                attributes=dict(meta.get("attributes", {})),
            )
        raise TypeError(f"Unsupported reference metadata entry: {meta!r}")


@dataclass
class RetrievalMatch:
    """A reference case returned by a retrieval query, with its score."""

    case: ReferenceCase
    similarity: float
