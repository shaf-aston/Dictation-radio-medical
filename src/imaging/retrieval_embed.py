"""DenseNet embedding extraction for the scan assistant.

Reuses the active classifier (lazy-loaded), freezes its backbone, and extracts
the L2-normalized feature vector before the final classification head. Kept in
its own module so importers that only index/search pre-computed embeddings do
not pull the torch-heavy extractor.
"""

from __future__ import annotations

import logging

import numpy as np

from src.imaging.classifier import get_active_classifier
from src.imaging.exceptions import ImagingError

logger = logging.getLogger(__name__)


class EmbeddingExtractor:
    """Extract normalized embeddings from chest X-rays via the DenseNet backbone.

    Loads the active classifier, freezes its backbone, and extracts the
    intermediate representation before the final classification head.
    """

    def __init__(self) -> None:
        """Initialize the extractor (no model load until first extract)."""
        self._classifier = None
        self._embedding_layer = None

    def _ensure_loaded(self) -> None:
        """Lazily load the classifier and freeze the backbone."""
        if self._classifier is not None:
            return
        self._classifier = get_active_classifier()
        model = self._classifier.model
        if model is None:
            raise ImagingError("Classifier model failed to load")
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        logger.info("Initialized embedding extractor with frozen backbone")

    def extract(self, image_path: str) -> np.ndarray:
        """Extract and normalize embedding from an image.

        Args:
            image_path: Path to the chest X-ray (PNG/JPG).

        Returns:
            A 1-D normalized (L2) numpy array of shape (embedding_dim,).

        Raises:
            ImagingError: If the image cannot be loaded or embedding extraction fails.
        """
        self._ensure_loaded()
        classifier = self._classifier
        assert classifier is not None

        try:
            import torch
            import torch.nn.functional as F
        except ImportError as exc:
            raise ImagingError("Embedding extraction requires torch") from exc

        try:
            # Preprocess the image using the classifier's pipeline
            tensor = classifier.preprocess(image_path)

            model = classifier.model
            with torch.no_grad():
                # For DenseNet, the final output before the classification head
                # is a feature vector. We'll hook the average pooling layer.
                # DenseNet121 has a global average pool -> dense layers.
                # Access features before the classifier head.
                features = model.features(tensor)  # (1, C, H, W)
                # Global average pooling
                embedding = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)  # (1, C)
                embedding = embedding[0].cpu().numpy()  # (C,)

            # Normalize to unit length
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding.astype(np.float32)
        except Exception as exc:
            raise ImagingError(f"Embedding extraction failed for {image_path}: {exc}") from exc
