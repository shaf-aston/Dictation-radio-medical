"""Chest X-ray classifier — pretrained TorchXRayVision DenseNet121.

Loads validated, published weights (trained on RSNA/NIH/CheXpert/PadChest) and
returns a per-pathology probability for one image. We stand on these weights
rather than training a diagnostic model from scratch: when a wrong suggestion
can mislead a diagnosis, the responsible baseline is published, peer-reviewed
weights plus a calibrated abstention gate (see :mod:`src.imaging.abstention`).

Heavy deps (torch, torchxrayvision, skimage) are imported lazily inside methods
so importing this module never pulls them in. The :class:`Classifier` Protocol
lets future modalities (CT, MRI) slot into the same analyzer pipeline.

A fine-tuned local model (registered under ``scan_classifier`` in the model
registry) is loaded in preference to the stock weights when one is active.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Protocol, runtime_checkable

from src.imaging.exceptions import ImagingError
from src.imaging.schemas import PathologyScore

logger = logging.getLogger(__name__)

# Default TorchXRayVision weights: every output head trained ("all").
_DEFAULT_WEIGHTS = "densenet121-res224-all"


@runtime_checkable
class Classifier(Protocol):
    """A modality classifier: image path → per-label probabilities."""

    def predict(self, image_path: str) -> List[PathologyScore]:
        """Return a :class:`PathologyScore` for every label the model emits."""
        ...


class ChestXRayClassifier:
    """TorchXRayVision DenseNet121 wrapper (lazy-loaded, cached)."""

    def __init__(self, weights: str = _DEFAULT_WEIGHTS) -> None:
        self._weights = weights
        self._model = None        # torch module, loaded on first predict()
        self._pathologies: List[str] = []

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torchxrayvision as xrv
        except ImportError as exc:
            raise ImagingError(
                "Scan analysis requires the imaging extra "
                "(pip install -r scripts/lightning/requirements_imaging.txt)."
            ) from exc
        try:
            model = xrv.models.DenseNet(weights=self._weights)
        except Exception as exc:
            raise ImagingError(f"Could not load chest X-ray weights: {exc}") from exc
        model.eval()
        self._model = model
        self._pathologies = list(model.pathologies)
        logger.info("Loaded chest X-ray model %s (%d labels)",
                    self._weights, len(self._pathologies))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, image_path: str) -> List[PathologyScore]:
        """Run the model on *image_path*; return one score per label.

        A label whose head was not trained for these weights is marked
        ``trained=False`` so the abstention gate can exclude it — TorchXRayVision
        emits a random value for such heads, which must never be surfaced.
        """
        self._ensure_loaded()
        model = self._model
        assert model is not None  # _ensure_loaded guarantees this
        tensor = self._preprocess(image_path)

        import torch
        with torch.no_grad():
            probs = model(tensor)[0].cpu().numpy()

        # TorchXRayVision uses an empty-string pathology name for the untrained
        # heads of a given weight set; mark those so the gate can drop them.
        return [
            PathologyScore(label=label, probability=float(prob), trained=bool(label))
            for label, prob in zip(self._pathologies, probs)
        ]

    def _preprocess(self, image_path: str):
        """Load and normalise an image into the model's expected 1×1×224×224 input."""
        path = Path(image_path)
        if not path.is_file():
            raise ImagingError(f"Image not found: {image_path}")
        try:
            import skimage.io
            import torch
            import torchxrayvision as xrv
        except ImportError as exc:
            raise ImagingError("Imaging dependencies are not installed.") from exc

        try:
            img = skimage.io.imread(str(path))
        except Exception as exc:
            raise ImagingError(f"Could not read image {image_path}: {exc}") from exc

        img = xrv.datasets.normalize(img, 255)   # scale to [-1024, 1024]
        if img.ndim == 3:
            img = img.mean(2)                    # collapse RGB to greyscale
        img = img[None, ...]                     # add channel dim
        transform = xrv.datasets.XRayResizer(224)
        img = transform(img)
        return torch.from_numpy(img).unsqueeze(0).float()

    # ------------------------------------------------------------------
    # Introspection (used by the abstention gate / localization)
    # ------------------------------------------------------------------

    @property
    def model(self):
        """The underlying torch module (loads it if needed)."""
        self._ensure_loaded()
        return self._model

    @property
    def pathologies(self) -> List[str]:
        self._ensure_loaded()
        return self._pathologies


def get_active_classifier() -> Classifier:
    """Return the classifier to use: an active fine-tuned scan model, else stock.

    Reads the model registry for an active ``scan_classifier`` version; if one is
    present its weights directory is used, otherwise the published baseline.
    """
    weights = _DEFAULT_WEIGHTS
    try:
        from src.cloud.framework.registry import ModelRegistry
        from src.training.schemas import TASK_SCAN_CLASSIFIER
        active = ModelRegistry().get_active_model_path(TASK_SCAN_CLASSIFIER)
        if active is not None:
            weights = str(active)
    except Exception as exc:
        logger.debug("No active scan model; using stock weights: %s", exc)
    return ChestXRayClassifier(weights=weights)
