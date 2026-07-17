"""Embedding-based image retrieval for the scan assistant (orchestration layer).

Builds and caches reference indices from labelled-image folders and answers
attribute-aware "find visually similar prior cases" queries for a query film.

The heavy pieces live in sibling modules and are re-exported here for back-compat:
  * :class:`~src.imaging.retrieval_embed.EmbeddingExtractor` — DenseNet embeddings.
  * :class:`~src.imaging.retrieval_index.EmbeddingIndex` — search + persistence.

Index and embeddings are cached to disk in ``data/cache/imaging_embeddings/``
(:func:`src.features.file_manager.imaging_embeddings_dir`) for reuse across
sessions.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from src.imaging.classifier import get_active_classifier  # noqa: F401 — re-exported; patch target
from src.imaging.datasets import DatasetRegistry, parse_label_file
from src.imaging.exceptions import ImagingError
from src.imaging.retrieval_embed import EmbeddingExtractor
from src.imaging.retrieval_index import EmbeddingIndex, Filters
from src.imaging.schemas import ReferenceCase, RetrievalMatch
from src.features.file_manager import imaging_embeddings_dir

__all__ = [
    "EmbeddingExtractor",
    "EmbeddingIndex",
    "Filters",
    "IndexBuilder",
    "retrieve_reference_cases",
    "get_reference_index",
]

logger = logging.getLogger(__name__)

# Subdirectory of the embeddings cache holding the persistent reference index
# built from every registered dataset (see :func:`get_reference_index`).
_REFERENCE_SUBDIR = "reference_index"


class IndexBuilder:
    """Build and cache embedding indices from labeled image folders."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        extractor: Optional[Any] = None,
    ) -> None:
        """Initialize the builder.

        Args:
            cache_dir: Directory for caching embeddings. Defaults to
                data/cache/imaging_embeddings/.
            extractor: Embedding extractor (anything exposing ``extract(path) ->
                np.ndarray``). Defaults to the real :class:`EmbeddingExtractor`;
                tests inject a lightweight fake to avoid loading torch.
        """
        if cache_dir is None:
            cache_dir = imaging_embeddings_dir()
        self._cache_dir = cache_dir
        self._extractor = extractor or EmbeddingExtractor()

    def build_from_folders(
        self,
        folder_paths: List[Path],
        label_suffix: str = ".json",
    ) -> EmbeddingIndex:
        """Build a single index by walking one or more labelled-image folders.

        Each PNG may have a sibling ``<name>.json`` sidecar (per-pathology labels
        plus an optional ``_attributes`` block). Images without a sidecar are
        still indexed for visual similarity, just with empty labels/attributes.

        Args:
            folder_paths: Directories containing PNG (+ optional JSON) pairs.
            label_suffix: Suffix for label files (default: ".json").

        Returns:
            A built EmbeddingIndex (not cached here — caching is the caller's
            choice; see :func:`get_reference_index`).

        Raises:
            ValueError: If no embeddings could be extracted from any folder.
        """
        embeddings_list: List[np.ndarray] = []
        metadata_list: List[dict] = []

        for folder_path in folder_paths:
            folder_path = Path(folder_path)
            if not folder_path.is_dir():
                logger.warning("Skipping missing reference folder: %s", folder_path)
                continue
            for png_path in sorted(folder_path.glob("*.png")):
                try:
                    emb = self._extractor.extract(str(png_path))
                except Exception as exc:
                    logger.warning("Failed to extract embedding for %s: %s", png_path, exc)
                    continue
                labels: dict = {}
                attributes: dict = {}
                sidecar = png_path.with_suffix(label_suffix)
                if sidecar.is_file():
                    try:
                        labels, attributes = parse_label_file(sidecar)
                    except (OSError, ValueError, json.JSONDecodeError) as exc:
                        logger.warning("Unreadable sidecar %s: %s", sidecar, exc)
                embeddings_list.append(emb)
                metadata_list.append(
                    ReferenceCase(str(png_path), labels, attributes).to_metadata())

        if not embeddings_list:
            raise ValueError(f"No embeddings extracted from {folder_paths}")

        index = EmbeddingIndex()
        index.build_index(embeddings_list, metadata_list)
        return index


def retrieve_reference_cases(
    query_embedding: np.ndarray,
    index: EmbeddingIndex,
    filters: Optional[Filters] = None,
    top_k: int = 5,
) -> List[RetrievalMatch]:
    """Retrieve the top-K reference cases for a query embedding, optionally filtered.

    The hybrid retrieval entry point: narrow the reference pool by attribute/label
    filters, then rank what remains by visual similarity to ``query_embedding``.

    Args:
        query_embedding: Normalized 1-D embedding of the query image (e.g. the one
            already cached on an :class:`~src.imaging.schemas.ImagingResult`).
        index: A built reference index (see :func:`get_reference_index`).
        filters: Optional attribute/label filter (see :data:`Filters`).
        top_k: Number of matches to return.

    Returns:
        A list of :class:`~src.imaging.schemas.RetrievalMatch`, similarity-desc.

    Raises:
        ImagingError: If the search fails.
    """
    try:
        hits = index.search(np.asarray(query_embedding), top_k=top_k, filters=filters)
        return [RetrievalMatch(ReferenceCase.from_metadata(meta), score) for score, meta in hits]
    except Exception as exc:
        raise ImagingError(f"Reference retrieval failed: {exc}") from exc


def _reference_cache_dir() -> Path:
    return imaging_embeddings_dir() / _REFERENCE_SUBDIR


def _reference_manifest_hash(registry: DatasetRegistry) -> str:
    """A stable fingerprint of the registered datasets' identities and sizes.

    Changes when a dataset is added/removed or its image count changes, which is
    the signal to rebuild the cached reference index.
    """
    parts: List[str] = []
    for name in sorted(registry.list_datasets()):
        try:
            count = len(registry.get_dataset_images(name))
            path = registry.get_dataset(name)["path"]
        except (KeyError, ValueError):
            continue
        parts.append(f"{name}:{path}:{count}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def get_reference_index(
    registry: Optional[DatasetRegistry] = None,
    extractor: Optional[Any] = None,
    force_rebuild: bool = False,
) -> Optional[EmbeddingIndex]:
    """Load (or build and cache) the reference index over all registered datasets.

    Built once and reused across sessions; rebuilt only when the set of registered
    datasets changes (tracked by a manifest hash). Returns ``None`` when no
    datasets are registered, so callers can degrade gracefully.

    Args:
        registry: Dataset registry to source reference folders from. Defaults to
            the on-disk :class:`DatasetRegistry`.
        extractor: Embedding extractor (test seam). Defaults to the real one.
        force_rebuild: Rebuild even if a fresh cache exists.

    Returns:
        A built :class:`EmbeddingIndex`, or ``None`` if there is no reference data.

    Raises:
        ImagingError: If building the index fails.
    """
    registry = registry or DatasetRegistry()
    names = registry.list_datasets()
    if not names:
        return None

    cache_dir = _reference_cache_dir()
    hash_file = cache_dir / "manifest_hash.txt"
    current_hash = _reference_manifest_hash(registry)

    if not force_rebuild and hash_file.is_file():
        try:
            if hash_file.read_text(encoding="utf-8").strip() == current_hash:
                index = EmbeddingIndex()
                index.load(cache_dir)
                return index
        except (OSError, FileNotFoundError, RuntimeError) as exc:
            logger.warning("Reference index cache unusable, rebuilding: %s", exc)

    folders: List[Path] = []
    for name in names:
        try:
            folders.append(Path(registry.get_dataset(name)["path"]))
        except (KeyError, ValueError):
            continue

    try:
        index = IndexBuilder(extractor=extractor).build_from_folders(folders)
        index.save(cache_dir)
        hash_file.write_text(current_hash, encoding="utf-8")
    except ValueError as exc:
        # No embeddings extractable (e.g. empty/imageless datasets) — treat as
        # "no reference data" rather than a hard failure.
        logger.info("No reference embeddings built: %s", exc)
        return None
    except Exception as exc:
        raise ImagingError(f"Failed to build reference index: {exc}") from exc

    return index
