"""Dataset registry for multi-book imaging training.

The DatasetRegistry tracks a collection of labelled chest X-ray datasets ("books")
registered on the local system. Each dataset is a directory of PNG images with
corresponding JSON label files.

Registry storage: ``data/imaging/datasets.json``
Schema: ``{datasets: {book_name: {path: str, metadata: dict, created_at: ISO8601}}}``

This registry allows the scan training task to aggregate examples from multiple
datasets when building a training batch — a radiologist might have labeled images
from different equipment, time periods, or patient populations, and the fine-tune
should learn from all of them.

Thread-safe: uses file locking (or a lock object) to serialize writes.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from src.features.file_manager import imaging_dir

logger = logging.getLogger(__name__)


class DatasetRegistry:
    """Registry of labelled chest X-ray datasets for fine-tuning.

    Manages a collection of named datasets, each pointing to a directory of
    PNG images with JSON label files. The registry is stored as JSON and
    provides thread-safe access.

    Example:
        >>> registry = DatasetRegistry()
        >>> registry.register_dataset("my_chest_xrays", "/path/to/images", {"patient_population": "trauma"})
        >>> registry.list_datasets()
        ["my_chest_xrays"]
        >>> dataset = registry.get_dataset("my_chest_xrays")
        >>> images = registry.get_dataset_images("my_chest_xrays")
    """

    def __init__(self, registry_path: Optional[Path] = None) -> None:
        """Initialize the registry.

        Args:
            registry_path: Path to the datasets.json file. Defaults to
                ``data/imaging/datasets.json``.
        """
        if registry_path is None:
            registry_path = imaging_dir() / "datasets.json"
        self._path = Path(registry_path)
        self._lock = threading.Lock()
        self._data: Dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        """Load the registry from disk, creating if missing."""
        with self._lock:
            if self._path.exists():
                try:
                    content = json.loads(self._path.read_text(encoding="utf-8"))
                    self._data = content.get("datasets", {})
                    logger.info("Loaded dataset registry with %d datasets", len(self._data))
                except (OSError, json.JSONDecodeError) as exc:
                    logger.warning("Failed to load dataset registry: %s, starting fresh", exc)
                    self._data = {}
            else:
                self._data = {}

    def _save(self) -> None:
        """Save the registry to disk."""
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            content = {"datasets": self._data}
            self._path.write_text(
                json.dumps(content, indent=2),
                encoding="utf-8"
            )
            logger.debug("Saved dataset registry to %s", self._path)

    def register_dataset(
        self,
        name: str,
        path: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        """Register or update a dataset.

        Args:
            name: Human-readable name for the dataset (book_name).
            path: Directory path containing PNG images and .json label files.
            metadata: Optional metadata dict (e.g., {"patient_population": "trauma"}).

        Raises:
            ValueError: If the path does not exist.
        """
        path_obj = Path(path)
        if not path_obj.is_dir():
            raise ValueError(f"Dataset path does not exist: {path}")

        with self._lock:
            self._data[name] = {
                "path": str(path_obj.absolute()),
                "metadata": metadata or {},
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        self._save()
        logger.info("Registered dataset '%s' at %s", name, path)

    def list_datasets(self) -> List[str]:
        """List all registered dataset names.

        Returns:
            A list of dataset names (book_names).
        """
        with self._lock:
            return list(self._data.keys())

    def get_dataset(self, name: str) -> dict:
        """Get a dataset entry.

        Args:
            name: The dataset name.

        Returns:
            A dict with keys: path, metadata, created_at.

        Raises:
            KeyError: If the dataset is not registered.
        """
        with self._lock:
            if name not in self._data:
                raise KeyError(f"Dataset not found: {name}")
            return dict(self._data[name])

    def get_dataset_images(self, name: str) -> List[str]:
        """Get all PNG image paths in a dataset.

        Walks the dataset directory and returns absolute paths to all *.png files.

        Args:
            name: The dataset name.

        Returns:
            A sorted list of absolute paths to PNG images.

        Raises:
            KeyError: If the dataset is not registered.
            ValueError: If the dataset path does not exist.
        """
        dataset = self.get_dataset(name)
        dataset_path = Path(dataset["path"])

        if not dataset_path.is_dir():
            raise ValueError(f"Dataset path no longer exists: {dataset_path}")

        return sorted([str(p.absolute()) for p in dataset_path.glob("*.png")])

    def remove_dataset(self, name: str) -> None:
        """Remove a dataset from the registry.

        Note: This does NOT delete the dataset folder, only the registry entry.

        Args:
            name: The dataset name.

        Raises:
            KeyError: If the dataset is not registered.
        """
        with self._lock:
            if name not in self._data:
                raise KeyError(f"Dataset not found: {name}")
            del self._data[name]
        self._save()
        logger.info("Removed dataset '%s' from registry", name)
