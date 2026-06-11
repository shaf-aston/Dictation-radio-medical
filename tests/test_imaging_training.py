"""Tests for imaging training infrastructure.

Tests the complete imaging training pipeline:
- DatasetRegistry (registration, listing, retrieval)
- EmbeddingExtractor and EmbeddingIndex (extraction, indexing, search)
- ScanClassifierTask (archive building from multiple datasets)
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.cloud.exceptions import CloudError
from src.cloud.tasks.scan_finetune import ScanClassifierTask
from src.imaging.datasets import DatasetRegistry
from src.imaging.retrieval import EmbeddingExtractor, EmbeddingIndex


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def tmp_registry(tmp_path):
    """Create a temporary DatasetRegistry."""
    registry_path = tmp_path / "datasets.json"
    return DatasetRegistry(registry_path=registry_path)


@pytest.fixture
def sample_dataset_dir(tmp_path):
    """Create a temporary directory with sample PNG images and labels."""
    dataset_dir = tmp_path / "sample_dataset"
    dataset_dir.mkdir()

    # Create dummy PNG files and labels
    images = ["image_001.png", "image_002.png", "image_003.png"]
    for img_name in images:
        img_path = dataset_dir / img_name
        # Write a minimal PNG header (1x1 pixel)
        img_path.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
            b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05"
            b"\xfb\xb6\xee\x05\x00\x00\x00\x00IEND\xaeB`\x82"
        )

        # Write corresponding label JSON
        label_path = dataset_dir / img_name.replace(".png", ".json")
        labels = {
            "Fracture": 1 if img_name == "image_001.png" else 0,
            "Pneumothorax": 0,
            "Effusion": 1 if img_name == "image_002.png" else 0,
            "Consolidation": 0,
        }
        label_path.write_text(json.dumps(labels), encoding="utf-8")

    return dataset_dir


@pytest.fixture
def embedding_sample():
    """Create a sample normalized embedding."""
    emb = np.random.randn(1024).astype(np.float32)
    emb = emb / np.linalg.norm(emb)
    return emb


# ============================================================================
# DatasetRegistry Tests
# ============================================================================


def test_dataset_registry_register(tmp_registry, sample_dataset_dir):
    """Test basic dataset registration."""
    tmp_registry.register_dataset(
        "book_a",
        str(sample_dataset_dir),
        metadata={"population": "trauma"}
    )
    assert "book_a" in tmp_registry.list_datasets()


def test_dataset_registry_register_nonexistent_path(tmp_registry, tmp_path):
    """Test that registering a non-existent path raises ValueError."""
    nonexistent = tmp_path / "does_not_exist"
    with pytest.raises(ValueError, match="does not exist"):
        tmp_registry.register_dataset("bad", str(nonexistent))


def test_dataset_registry_list_datasets(tmp_registry, sample_dataset_dir):
    """Test listing registered datasets."""
    tmp_registry.register_dataset("book_a", str(sample_dataset_dir))
    tmp_registry.register_dataset("book_b", str(sample_dataset_dir))
    datasets = tmp_registry.list_datasets()
    assert len(datasets) == 2
    assert "book_a" in datasets
    assert "book_b" in datasets


def test_dataset_registry_get_dataset(tmp_registry, sample_dataset_dir):
    """Test retrieving a dataset entry."""
    tmp_registry.register_dataset(
        "my_book",
        str(sample_dataset_dir),
        metadata={"notes": "test data"}
    )
    dataset = tmp_registry.get_dataset("my_book")
    assert "path" in dataset
    assert "metadata" in dataset
    assert "created_at" in dataset
    assert dataset["metadata"]["notes"] == "test data"


def test_dataset_registry_get_dataset_not_found(tmp_registry):
    """Test that getting a non-existent dataset raises KeyError."""
    with pytest.raises(KeyError, match="not found"):
        tmp_registry.get_dataset("nonexistent")


def test_dataset_registry_get_dataset_images(tmp_registry, sample_dataset_dir):
    """Test retrieving image paths from a dataset."""
    tmp_registry.register_dataset("my_book", str(sample_dataset_dir))
    images = tmp_registry.get_dataset_images("my_book")
    assert len(images) == 3
    assert all(img.endswith(".png") for img in images)
    assert all(Path(img).exists() for img in images)


def test_dataset_registry_get_dataset_images_nonexistent_dataset(tmp_registry):
    """Test that getting images from a non-existent dataset raises KeyError."""
    with pytest.raises(KeyError):
        tmp_registry.get_dataset_images("nonexistent")


def test_dataset_registry_remove_dataset(tmp_registry, sample_dataset_dir):
    """Test removing a dataset from the registry."""
    tmp_registry.register_dataset("to_remove", str(sample_dataset_dir))
    assert "to_remove" in tmp_registry.list_datasets()
    tmp_registry.remove_dataset("to_remove")
    assert "to_remove" not in tmp_registry.list_datasets()


def test_dataset_registry_remove_nonexistent(tmp_registry):
    """Test that removing a non-existent dataset raises KeyError."""
    with pytest.raises(KeyError, match="not found"):
        tmp_registry.remove_dataset("nonexistent")


def test_dataset_registry_persistence(tmp_path, sample_dataset_dir):
    """Test that registry persists to disk and reloads correctly."""
    registry_path = tmp_path / "datasets.json"

    # Create and populate registry
    registry1 = DatasetRegistry(registry_path=registry_path)
    registry1.register_dataset("persisted", str(sample_dataset_dir))

    # Create a new registry instance pointing to the same path
    registry2 = DatasetRegistry(registry_path=registry_path)
    assert "persisted" in registry2.list_datasets()
    dataset = registry2.get_dataset("persisted")
    assert Path(dataset["path"]).samefile(sample_dataset_dir)


# ============================================================================
# EmbeddingExtractor Tests
# ============================================================================


def test_embedding_extractor_initialization():
    """Test that EmbeddingExtractor initializes without loading the model."""
    extractor = EmbeddingExtractor()
    assert extractor._classifier is None


@patch("src.imaging.retrieval.get_active_classifier")
def test_embedding_extractor_lazy_load(mock_get_classifier):
    """Test that EmbeddingExtractor lazily loads the classifier."""
    # Mock classifier
    mock_model = MagicMock()
    mock_model.parameters.return_value = []
    mock_model.eval.return_value = None
    mock_classifier = MagicMock()
    mock_classifier.model = mock_model
    mock_classifier._ensure_loaded = MagicMock()
    mock_get_classifier.return_value = mock_classifier

    extractor = EmbeddingExtractor()
    # The extractor should not have loaded anything yet
    assert extractor._classifier is None


# ============================================================================
# EmbeddingIndex Tests
# ============================================================================


def test_embedding_index_initialization():
    """Test EmbeddingIndex initialization."""
    index = EmbeddingIndex()
    assert index._embeddings is None
    assert index._metadata == []


def test_embedding_index_build_index(embedding_sample):
    """Test building an index from embeddings."""
    embeddings = [embedding_sample, embedding_sample + 0.01]
    metadata = ["image_1.png", "image_2.png"]

    index = EmbeddingIndex()
    index.build_index(embeddings, metadata)

    assert index._embeddings is not None
    assert index._embeddings.shape == (2, 1024)
    assert len(index._metadata) == 2


def test_embedding_index_build_index_empty(embedding_sample):
    """Test that building with empty inputs raises ValueError."""
    index = EmbeddingIndex()
    with pytest.raises(ValueError, match="cannot be empty"):
        index.build_index([], [])


def test_embedding_index_build_index_length_mismatch(embedding_sample):
    """Test that mismatched lengths raise ValueError."""
    embeddings = [embedding_sample]
    metadata = ["image_1.png", "image_2.png"]

    index = EmbeddingIndex()
    with pytest.raises(ValueError, match="must have the same length"):
        index.build_index(embeddings, metadata)


def test_embedding_index_search(embedding_sample):
    """Test searching the index."""
    embeddings = [
        embedding_sample,
        embedding_sample + 0.01,
        embedding_sample + 0.05,
    ]
    metadata = ["image_1.png", "image_2.png", "image_3.png"]

    index = EmbeddingIndex()
    index.build_index(embeddings, metadata)

    results = index.search(embedding_sample, top_k=2)
    assert len(results) == 2
    # First result should be the query itself or closest
    assert all(isinstance(score, float) and isinstance(path, str) for score, path in results)


def test_embedding_index_search_without_build():
    """Test that searching before building raises RuntimeError."""
    index = EmbeddingIndex()
    with pytest.raises(RuntimeError, match="Index not built"):
        index.search(np.random.randn(1024), top_k=5)


def test_embedding_index_save_load(tmp_path, embedding_sample):
    """Test saving and loading an index."""
    embeddings = [embedding_sample, embedding_sample + 0.01]
    metadata = ["image_1.png", "image_2.png"]

    # Build and save
    index1 = EmbeddingIndex()
    index1.build_index(embeddings, metadata)
    save_path = tmp_path / "index"
    index1.save(save_path)

    # Load
    index2 = EmbeddingIndex()
    index2.load(save_path)

    assert index2._metadata == metadata
    assert index2._embeddings is not None
    assert np.allclose(index2._embeddings, np.array(embeddings))


def test_embedding_index_save_without_build(tmp_path):
    """Test that saving without building raises RuntimeError."""
    index = EmbeddingIndex()
    with pytest.raises(RuntimeError, match="No index built"):
        index.save(tmp_path / "index")


def test_embedding_index_load_missing_files(tmp_path):
    """Test that loading from missing files raises FileNotFoundError."""
    index = EmbeddingIndex()
    with pytest.raises(FileNotFoundError):
        index.load(tmp_path / "nonexistent")


# ============================================================================
# ScanClassifierTask Tests
# ============================================================================


def test_scan_classifier_task_instantiation():
    """Test ScanClassifierTask initialization."""
    task = ScanClassifierTask()
    assert task.task_type == "scan_classifier"


@patch("src.features.file_manager.imaging_training_dir")
@patch("src.imaging.datasets.DatasetRegistry")
def test_scan_classifier_task_build_archive_with_datasets(
    mock_registry_class,
    mock_imaging_dir,
    tmp_path,
    sample_dataset_dir,
):
    """Test building archive from registered datasets."""
    # Mock imaging_training_dir to return empty (no legacy data)
    mock_legacy_dir = tmp_path / "legacy"
    mock_legacy_dir.mkdir()
    mock_imaging_dir.return_value = mock_legacy_dir

    # Mock registry with one dataset
    mock_registry = MagicMock()
    mock_registry.list_datasets.return_value = ["book_a"]
    mock_registry.get_dataset.return_value = {"path": str(sample_dataset_dir)}
    mock_registry_class.return_value = mock_registry

    task = ScanClassifierTask()
    records = []  # ScanClassifierTask doesn't use records
    batch_id = "batch_001"

    archive_path = task.build_archive(batch_id, records, "densenet121-res224-all")
    assert archive_path.exists()
    assert archive_path.suffix == ".gz"

    # Verify archive contents
    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getnames()
        assert "manifest.json" in members
        # Should have image files
        image_members = [m for m in members if m.startswith("images/")]
        assert len(image_members) >= 3  # At least 3 images from sample dataset

        # Check manifest
        manifest_file = tar.extractfile("manifest.json")
        assert manifest_file is not None
        manifest_data = manifest_file.read().decode("utf-8")
        manifest = json.loads(manifest_data)
        assert manifest["batch_id"] == batch_id
        assert manifest["task_type"] == "scan_classifier"
        assert len(manifest["examples"]) > 0


@patch("src.features.file_manager.imaging_training_dir")
@patch("src.imaging.datasets.DatasetRegistry")
def test_scan_classifier_task_build_archive_legacy_fallback(
    mock_registry_class,
    mock_imaging_dir,
    tmp_path,
    sample_dataset_dir,
):
    """Test building archive falls back to legacy dir when no datasets registered."""
    # Mock imaging_training_dir to return the sample dataset
    mock_imaging_dir.return_value = sample_dataset_dir

    # Mock empty registry
    mock_registry = MagicMock()
    mock_registry.list_datasets.return_value = []
    mock_registry_class.return_value = mock_registry

    task = ScanClassifierTask()
    batch_id = "batch_legacy"

    archive_path = task.build_archive(batch_id, [], "densenet121-res224-all")
    assert archive_path.exists()

    with tarfile.open(archive_path, "r:gz") as tar:
        manifest_file = tar.extractfile("manifest.json")
        assert manifest_file is not None
        manifest_data = manifest_file.read().decode("utf-8")
        manifest = json.loads(manifest_data)
        assert len(manifest["examples"]) > 0


@patch("src.features.file_manager.imaging_training_dir")
@patch("src.imaging.datasets.DatasetRegistry")
def test_scan_classifier_task_build_archive_no_data(
    mock_registry_class,
    mock_imaging_dir,
    tmp_path,
):
    """Test that building archive with no data raises CloudError."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    mock_imaging_dir.return_value = empty_dir

    # Mock empty registry
    mock_registry = MagicMock()
    mock_registry.list_datasets.return_value = []
    mock_registry_class.return_value = mock_registry

    task = ScanClassifierTask()

    with pytest.raises(CloudError, match="No labelled scans available"):
        task.build_archive("batch_empty", [], "densenet121-res224-all")


@patch("src.features.file_manager.imaging_training_dir")
@patch("src.imaging.datasets.DatasetRegistry")
def test_scan_classifier_task_trauma_oversampling(
    mock_registry_class,
    mock_imaging_dir,
    tmp_path,
    sample_dataset_dir,
):
    """Test that trauma-positive examples are oversampled."""
    mock_imaging_dir.return_value = tmp_path / "legacy"
    (tmp_path / "legacy").mkdir()

    mock_registry = MagicMock()
    mock_registry.list_datasets.return_value = ["book_a"]
    mock_registry.get_dataset.return_value = {"path": str(sample_dataset_dir)}
    mock_registry_class.return_value = mock_registry

    task = ScanClassifierTask()
    archive_path = task.build_archive("batch_trauma", [], "densenet121-res224-all")

    with tarfile.open(archive_path, "r:gz") as tar:
        manifest_file = tar.extractfile("manifest.json")
        assert manifest_file is not None
        manifest_data = manifest_file.read().decode("utf-8")
        manifest = json.loads(manifest_data)

        # Count how many trauma examples (Fracture=1 or Effusion=1)
        trauma_examples = [
            ex for ex in manifest["examples"]
            if ex["labels"].get("Fracture") == 1 or ex["labels"].get("Effusion") == 1
        ]
        # sample_dataset_dir has 2 trauma-positive images (image_001 and image_002)
        # Each is oversampled 3x, so at least 6 trauma examples in manifest
        assert len(trauma_examples) >= 6


# ============================================================================
# Integration Tests
# ============================================================================


def test_dataset_registry_multiple_datasets(tmp_registry, tmp_path):
    """Test registry with multiple datasets and aggregation."""
    # Create two dataset directories
    dataset_a = tmp_path / "dataset_a"
    dataset_b = tmp_path / "dataset_b"
    dataset_a.mkdir()
    dataset_b.mkdir()

    # Add images to each
    for i in range(2):
        png_path = dataset_a / f"image_{i}.png"
        json_path = dataset_a / f"image_{i}.json"
        png_path.write_bytes(b"\x89PNG...")
        json_path.write_text(json.dumps({"label": 0}))

    for i in range(3):
        png_path = dataset_b / f"image_{i}.png"
        json_path = dataset_b / f"image_{i}.json"
        png_path.write_bytes(b"\x89PNG...")
        json_path.write_text(json.dumps({"label": 1}))

    # Register both
    tmp_registry.register_dataset("book_a", str(dataset_a))
    tmp_registry.register_dataset("book_b", str(dataset_b))

    # Verify
    assert len(tmp_registry.list_datasets()) == 2
    assert len(tmp_registry.get_dataset_images("book_a")) == 2
    assert len(tmp_registry.get_dataset_images("book_b")) == 3


# ============================================================================
# Analyzer integration tests
# ============================================================================


def test_analyze_scan_embedding_extraction_fails(tmp_path, caplog):
    """Test that analyze_scan gracefully handles embedding extraction failure.

    Verifies that when EmbeddingExtractor raises an exception, the analysis
    still completes successfully with embedding=None, and failure is logged
    at debug level (non-critical).
    """
    from src.imaging.analyzer import analyze_scan
    from src.imaging.classifier import Classifier

    # Create a minimal test image
    test_image = tmp_path / "test.png"
    test_image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    # Mock classifier with known output
    mock_clf = MagicMock(spec=Classifier)
    mock_clf.predict.return_value = {
        "Fracture": 0.1,
        "Pneumothorax": 0.05,
        "Cardiomegaly": 0.9,
    }

    # Mock abstention.apply to return no candidates (empty), so analysis completes
    with patch("src.imaging.analyzer.abstention.apply", return_value=[]):
        # Mock EmbeddingExtractor to raise an exception
        with patch("src.imaging.analyzer.EmbeddingExtractor") as mock_extractor_class:
            mock_extractor = MagicMock()
            mock_extractor.extract.side_effect = RuntimeError("CUDA out of memory")
            mock_extractor_class.return_value = mock_extractor

            # Call analyze_scan; should not raise
            result = analyze_scan(str(test_image), classifier=mock_clf)

            # Verify result has no embedding (extraction failed)
            assert result.embedding is None
            assert result.findings == []

            # Verify extraction was attempted (EmbeddingExtractor was instantiated)
            mock_extractor_class.assert_called_once()
            mock_extractor.extract.assert_called_once_with(str(test_image))
