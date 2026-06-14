"""Scan-assistant orchestrator — image in, reviewed suggestions out.

This is the public entry point and the only file the UI calls. Like
``dictation/postprocess/pipeline.py`` it just *sequences* the stages, each of
which owns one responsibility:

    classifier  → raw per-label probabilities
    abstention  → keep only trained, confident labels (the safety gate)
    localization→ a Grad-CAM region per kept finding
    embedding   → normalized embedding for similarity search (cached in result)
    (here)      → drop any finding without a region, render the overlay

The hard safety invariant lives here: **a finding is returned to the user only
if it is confident AND has a localised region.** A confident finding we cannot
point to is withheld, because an unexplained suggestion is more dangerous than
silence. Every result carries the standard non-diagnostic disclaimer.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional, TYPE_CHECKING

from src.imaging import abstention, localization
from src.imaging.classifier import Classifier, get_active_classifier
from src.imaging.schemas import ImagingResult

if TYPE_CHECKING:
    from src.imaging.retrieval import EmbeddingExtractor

logger = logging.getLogger(__name__)

# Optional import for embedding extraction (lazy-loaded, can fail gracefully)
_EMBEDDING_AVAILABLE = False
try:
    from src.imaging.retrieval import EmbeddingExtractor
    _EMBEDDING_AVAILABLE = True
except ImportError:
    pass


def analyze_scan(
    image_path: str, classifier: Optional[Classifier] = None
) -> ImagingResult:
    """Analyse one chest X-ray and return reviewed, localised suggestions.

    Args:
        image_path: Path to the image (PNG/JPG/DICOM-exported greyscale).
        classifier: Optional injected classifier (tests pass a fake); defaults to
            the active fine-tuned scan model, else the published baseline.

    Returns:
        An :class:`ImagingResult`. ``findings`` contains only confident, localised
        suggestions; ``all_scores`` retains every raw score for audit/debugging;
        ``embedding`` caches the normalized embedding for retrieval/caching;
        ``disclaimer`` is always populated. Raises :class:`ImagingError` only when
        the image itself cannot be classified.
    """
    clf = classifier or get_active_classifier()

    scores = clf.predict(image_path)            # may raise ImagingError
    candidates = abstention.apply(scores)        # safety gate: confident + trained

    # Extract embedding for caching/retrieval (best-effort; failure is not critical)
    embedding = None
    if _EMBEDDING_AVAILABLE:
        try:
            extractor = EmbeddingExtractor()
            embedding = extractor.extract(image_path)
        except Exception as exc:
            logger.debug("Embedding extraction failed (non-critical): %s", exc)

    if not candidates:
        logger.info("No findings cleared the abstention gate for %s", image_path)
        return ImagingResult(findings=[], all_scores=scores, embedding=embedding)

    regions = localization.localize(clf, image_path, candidates)
    # Enforce the invariant: keep only findings we can point to.
    confirmed = []
    for f in candidates:
        region = regions.get(f.label)
        if region is None:
            logger.info("Withholding '%s' — confident but not localised", f.label)
            continue
        f.region = region
        confirmed.append(f)

    overlay_path = None
    if confirmed:
        from src.features.file_manager import imaging_overlay_dir
        out = imaging_overlay_dir() / f"overlay_{uuid.uuid4().hex[:8]}.png"
        overlay_path = localization.render_overlay(image_path, confirmed, out)

    return ImagingResult(
        findings=confirmed,
        overlay_png_path=overlay_path,
        all_scores=scores,
        embedding=embedding,
    )
