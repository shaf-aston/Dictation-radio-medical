"""In-memory embedding index with attribute-aware similarity search + persistence.

Holds normalized embeddings plus their structured metadata and answers top-K
similarity queries — via HNSW (hnswlib) when available, else a numpy brute-force
dot product. Split from the orchestration layer so the index can be built and
searched (e.g. in tests) without the torch-heavy extractor.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.core.json_store import read_json, write_json
from src.imaging.schemas import ReferenceCase

logger = logging.getLogger(__name__)

# A retrieval filter is a plain dict with any of these optional keys:
#   required_labels: Iterable[str] — keep cases positive (==1) for ANY listed label
#   view:            str           — require attributes["view"] to match (case-insensitive)
#   age_range:       (min, max)    — require attributes["age"] within the inclusive range
Filters = Dict[str, Any]


class EmbeddingIndex:
    """Index normalized embeddings for fast similarity search.

    Uses HNSW (Hierarchical Navigable Small World) via hnswlib if available,
    otherwise falls back to brute-force cosine similarity via numpy.
    """

    def __init__(self) -> None:
        """Initialize an empty index."""
        self._embeddings: Optional[np.ndarray] = None
        self._metadata: List[Dict[str, Any]] = []
        self._hnsw_index = None
        self._use_hnsw = False

    def build_index(
        self,
        embeddings_list: List[np.ndarray],
        metadata: List[Dict[str, Any]],
    ) -> None:
        """Build an index from embeddings and structured metadata.

        Args:
            embeddings_list: List of normalized (1-D) embeddings.
            metadata: One metadata dict per embedding (``{"path", "labels",
                "attributes"}`` — see :meth:`ReferenceCase.to_metadata`). For
                backward compatibility a bare path string is also accepted and
                wrapped into an attribute-less case.

        Raises:
            ValueError: If inputs are empty or lengths don't match.
        """
        if not embeddings_list or not metadata:
            raise ValueError("Embeddings and metadata lists cannot be empty")
        if len(embeddings_list) != len(metadata):
            raise ValueError("Embeddings and metadata must have the same length")

        # Stack embeddings into a 2-D array
        self._embeddings = np.array(embeddings_list, dtype=np.float32)
        # Normalise every entry to the structured dict form (upgrades legacy
        # bare-path strings) so search/save can assume a consistent shape.
        self._metadata = [ReferenceCase.from_metadata(m).to_metadata() for m in metadata]

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

    @staticmethod
    def _matches_filters(meta: Dict[str, Any], filters: Filters) -> bool:
        """True if a metadata entry satisfies every active filter clause.

        Standard filter semantics: a clause that references an attribute the case
        does not carry excludes that case (so legacy attribute-less data is only
        ever returned when no attribute filter is active).
        """
        required = filters.get("required_labels")
        if required:
            labels = meta.get("labels", {})
            if not any(labels.get(name) == 1 for name in required):
                return False

        attrs = meta.get("attributes", {})

        view = filters.get("view")
        if view is not None:
            case_view = attrs.get("view")
            if case_view is None or str(case_view).lower() != str(view).lower():
                return False

        age_range = filters.get("age_range")
        if age_range is not None:
            age = attrs.get("age")
            if age is None or not (age_range[0] <= age <= age_range[1]):
                return False

        return True

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filters: Optional[Filters] = None,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """Search for the top-K most similar embeddings, optionally filtered.

        Args:
            query_embedding: A normalized 1-D embedding.
            top_k: Number of results to return.
            filters: Optional attribute/label filter (see :data:`Filters`). When
                present, the candidate pool is narrowed first and ranked by
                similarity within it (brute-force over the subset). When absent,
                the fast HNSW path is used if available.

        Returns:
            A list of (similarity_score, metadata-dict) tuples, sorted by
            similarity (descending).

        Raises:
            RuntimeError: If the index has not been built.
        """
        if self._embeddings is None or not self._metadata:
            raise RuntimeError("Index not built; call build_index first")

        query_embedding = np.array(query_embedding, dtype=np.float32)

        if filters:
            candidate_idx = np.array(
                [i for i, m in enumerate(self._metadata) if self._matches_filters(m, filters)],
                dtype=np.int64,
            )
            if candidate_idx.size == 0:
                return []
            sims = self._embeddings[candidate_idx] @ query_embedding
            order = np.argsort(sims)[-top_k:][::-1]
            return [(float(sims[j]), self._metadata[int(candidate_idx[j])]) for j in order]

        if self._use_hnsw and self._hnsw_index is not None:
            # HNSW search (cosine distance = 1 - similarity)
            k = min(top_k, len(self._metadata))
            indices, distances = self._hnsw_index.knn_query(query_embedding[None, :], k=k)
            return [
                (1.0 - float(dist), self._metadata[idx])
                for idx, dist in zip(indices[0], distances[0])
            ]

        # Brute-force cosine similarity (normalized vectors → dot product)
        similarities = np.dot(self._embeddings, query_embedding)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(float(similarities[idx]), self._metadata[idx]) for idx in top_indices]

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
        write_json(path / "metadata.json", self._metadata)
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
        raw_metadata = read_json(json_path, [])
        # Normalise (upgrades legacy bare-path-string indices to structured dicts).
        self._metadata = [ReferenceCase.from_metadata(m).to_metadata() for m in raw_metadata]
        logger.info("Loaded index from %s (%d embeddings)", path, len(self._metadata))
