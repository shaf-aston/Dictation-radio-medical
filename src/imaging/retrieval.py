"""Embedding-based image retrieval for the scan assistant.

Extracts normalized embeddings from chest X-rays using the DenseNet backbone
and indexes them for similarity search. This enables the system to find visually
similar historical cases given a query image.

The embedding extractor reuses the active classifier (lazy-loaded) and freezes
its backbone; embeddings are extracted from the final dense layer before
classification. An optional HNSW index (via hnswlib) provides fast approximate
nearest-neighbor search; falls back to brute-force cosine similarity with scipy
if HNSW is unavailable.

Index and embeddings are cached to disk in ``data/imaging/embeddings/`` for
reuse across sessions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.imaging.exceptions import ImagingError
from src.imaging.classifier import get_active_classifier
from src.features.file_manager import imaging_dir

logger = logging.getLogger(__name__)

# Namespace for embedding caches within data/imaging/
_EMBEDDINGS_SUBDIR = "embeddings"


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
        self._classifier._ensure_loaded()  # type: ignore[attr-defined]
        model = self._classifier.model  # type: ignore[attr-defined]
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
            tensor = classifier._preprocess(image_path)  # type: ignore[attr-defined]

            model = classifier.model  # type: ignore[attr-defined]
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


class EmbeddingIndex:
    """Index normalized embeddings for fast similarity search.

    Uses HNSW (Hierarchical Navigable Small World) via hnswlib if available,
    otherwise falls back to brute-force cosine similarity via scipy.
    """

    def __init__(self) -> None:
        """Initialize an empty index."""
        self._embeddings: Optional[np.ndarray] = None
        self._metadata: List[str] = []
        self._hnsw_index = None
        self._use_hnsw = False

    def build_index(
        self,
        embeddings_list: List[np.ndarray],
        metadata: List[str],
    ) -> None:
        """Build an index from embeddings and metadata.

        Args:
            embeddings_list: List of normalized (1-D) embeddings.
            metadata: List of metadata strings (e.g., image paths), same length.

        Raises:
            ValueError: If inputs are empty or lengths don't match.
        """
        if not embeddings_list or not metadata:
            raise ValueError("Embeddings and metadata lists cannot be empty")
        if len(embeddings_list) != len(metadata):
            raise ValueError("Embeddings and metadata must have the same length")

        # Stack embeddings into a 2-D array
        self._embeddings = np.array(embeddings_list, dtype=np.float32)
        self._metadata = list(metadata)

        # Try to build HNSW index
        try:
            import hnswlib
            emb_dim = self._embeddings.shape[1]
            self._hnsw_index = hnswlib.Index(space="cosine", dim=emb_dim)
            self._hnsw_index.init_index(max_elements=len(embeddings_list), ef_construction=200, M=16)
            self._hnsw_index.add_items(self._embeddings, np.arange(len(embeddings_list)))
            self._hnsw_index.set_ef(50)
            self._use_hnsw = True
            logger.info("Built HNSW index for %d embeddings", len(embeddings_list))
        except ImportError:
            logger.info("hnswlib not available; using brute-force cosine similarity")
            self._use_hnsw = False
        except Exception as exc:
            logger.warning("HNSW index build failed, falling back to brute-force: %s", exc)
            self._use_hnsw = False

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[Tuple[float, str]]:
        """Search for the top-K most similar embeddings.

        Args:
            query_embedding: A normalized 1-D embedding.
            top_k: Number of results to return.

        Returns:
            A list of (similarity_score, metadata) tuples, sorted by similarity (desc).

        Raises:
            RuntimeError: If the index has not been built.
        """
        if self._embeddings is None or not self._metadata:
            raise RuntimeError("Index not built; call build_index first")

        query_embedding = np.array(query_embedding, dtype=np.float32)

        if self._use_hnsw and self._hnsw_index is not None:
            # HNSW search
            indices, distances = self._hnsw_index.knn_query(query_embedding[None, :], k=top_k)
            # distances from HNSW cosine are (1 - similarity); convert back
            return [
                (1.0 - float(dist), self._metadata[idx])
                for idx, dist in zip(indices[0], distances[0])
            ]
        else:
            # Brute-force cosine similarity using numpy (normalized embeddings)
            # For normalized vectors, cosine similarity = dot product
            similarities = np.dot(self._embeddings, query_embedding)
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            return [
                (float(similarities[idx]), self._metadata[idx])
                for idx in top_indices
            ]

    def save(self, path: Path) -> None:
        """Save index and embeddings to disk.

        Args:
            path: Directory to save to (will be created if needed).
        """
        path.mkdir(parents=True, exist_ok=True)
        if self._embeddings is None:
            raise RuntimeError("No index built; nothing to save")

        # Save embeddings as NPZ
        np.savez_compressed(
            str(path / "embeddings.npz"),
            embeddings=self._embeddings,
        )
        # Save metadata as JSON
        with open(path / "metadata.json", "w") as f:
            json.dump(self._metadata, f)
        logger.info("Saved index to %s", path)

    def load(self, path: Path) -> None:
        """Load index and embeddings from disk.

        Args:
            path: Directory to load from.

        Raises:
            FileNotFoundError: If required files are missing.
        """
        if not path.exists():
            raise FileNotFoundError(f"Index path does not exist: {path}")

        # Load embeddings
        npz_path = path / "embeddings.npz"
        json_path = path / "metadata.json"
        if not npz_path.exists() or not json_path.exists():
            raise FileNotFoundError(f"Missing index files in {path}")

        data = np.load(str(npz_path))
        self._embeddings = data["embeddings"]
        with open(json_path, "r") as f:
            self._metadata = json.load(f)
        logger.info("Loaded index from %s (%d embeddings)", path, len(self._metadata))


class IndexBuilder:
    """Build and cache embedding indices from labeled image folders."""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        """Initialize the builder with an optional cache directory.

        Args:
            cache_dir: Directory for caching embeddings. Defaults to
                data/imaging/embeddings/.
        """
        if cache_dir is None:
            cache_dir = imaging_dir() / _EMBEDDINGS_SUBDIR
        self._cache_dir = cache_dir
        self._extractor = EmbeddingExtractor()

    def build_from_folder(
        self,
        folder_path: Path,
        label_suffix: str = ".json",
    ) -> EmbeddingIndex:
        """Build an index by walking a folder of labeled images.

        Expects each PNG to have a corresponding JSON file (default: ``<name>.json``)
        containing labels.

        Args:
            folder_path: Directory containing PNG + JSON pairs.
            label_suffix: Suffix for label files (default: ".json").

        Returns:
            A built EmbeddingIndex, with embeddings cached to disk.

        Raises:
            ValueError: If no images are found.
        """
        folder_path = Path(folder_path)
        if not folder_path.is_dir():
            raise ValueError(f"Folder does not exist: {folder_path}")

        # Find all PNG files
        pngs = list(folder_path.glob("*.png"))
        if not pngs:
            raise ValueError(f"No PNG files found in {folder_path}")

        embeddings_list: List[np.ndarray] = []
        metadata_list: List[str] = []

        for png_path in sorted(pngs):
            try:
                emb = self._extractor.extract(str(png_path))
                embeddings_list.append(emb)
                metadata_list.append(str(png_path))
                logger.debug("Extracted embedding for %s", png_path.name)
            except Exception as exc:
                logger.warning("Failed to extract embedding for %s: %s", png_path, exc)
                continue

        if not embeddings_list:
            raise ValueError(f"No embeddings extracted from {folder_path}")

        # Build index
        index = EmbeddingIndex()
        index.build_index(embeddings_list, metadata_list)

        # Cache to disk
        folder_hash = hash(str(folder_path)) & 0xffffffff
        cache_subdir = self._cache_dir / f"folder_{folder_hash:08x}"
        try:
            index.save(cache_subdir)
        except Exception as exc:
            logger.warning("Failed to cache index: %s", exc)

        return index


def retrieve_similar_images(
    query_image_path: str,
    reference_folder: str,
    top_k: int = 5,
) -> List[Tuple[float, str]]:
    """Retrieve the top-K most similar images from a reference folder.

    Args:
        query_image_path: Path to the query chest X-ray.
        reference_folder: Path to folder containing reference images.
        top_k: Number of results to return.

    Returns:
        A list of (similarity_score, reference_image_path) tuples, sorted
        by similarity (descending).

    Raises:
        ImagingError: If embeddings cannot be extracted or index cannot be built.
    """
    try:
        extractor = EmbeddingExtractor()
        query_emb = extractor.extract(query_image_path)

        builder = IndexBuilder()
        index = builder.build_from_folder(Path(reference_folder))

        return index.search(query_emb, top_k=top_k)
    except Exception as exc:
        raise ImagingError(f"Image retrieval failed: {exc}") from exc
