"""Tests for attribute-aware image retrieval.

Runs without torch: every test injects tiny fake embeddings (via a fake
extractor or by calling ``EmbeddingIndex.build_index`` directly), so the real
DenseNet backbone is never loaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.imaging.datasets import DatasetRegistry, parse_label_file
from src.imaging.retrieval import (
    EmbeddingIndex,
    get_reference_index,
    retrieve_reference_cases,
)
from src.imaging.schemas import ReferenceCase


def _unit(vec) -> np.ndarray:
    """Return a float32 L2-normalised copy of *vec*."""
    arr = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return arr / norm if norm else arr


class _FakeExtractor:
    """Maps an image filename to a preset embedding — no torch, no file read."""

    def __init__(self, vectors: dict) -> None:
        self.vectors = {k: _unit(v) for k, v in vectors.items()}
        self.calls: list = []

    def extract(self, path: str) -> np.ndarray:
        self.calls.append(path)
        return self.vectors[Path(path).name]


class _RaisingExtractor:
    """Fails if asked to extract — proves the cached index path skips extraction."""

    def extract(self, path: str) -> np.ndarray:  # pragma: no cover - must not run
        raise AssertionError(f"extract() should not be called for {path}")


# --------------------------------------------------------------------------- #
# parse_label_file
# --------------------------------------------------------------------------- #

def test_parse_label_file_flat_legacy(tmp_path: Path) -> None:
    sidecar = tmp_path / "img.json"
    sidecar.write_text(json.dumps({"Fracture": 1, "Effusion": 0}))
    labels, attributes = parse_label_file(sidecar)
    assert labels == {"Fracture": 1, "Effusion": 0}
    assert attributes == {}


def test_parse_label_file_with_attributes(tmp_path: Path) -> None:
    sidecar = tmp_path / "img.json"
    sidecar.write_text(json.dumps(
        {"Fracture": 1, "_attributes": {"age": 54, "view": "PA", "bmi": 27.1}}))
    labels, attributes = parse_label_file(sidecar)
    assert labels == {"Fracture": 1}                # _attributes excluded from labels
    assert attributes == {"age": 54, "view": "PA", "bmi": 27.1}


# --------------------------------------------------------------------------- #
# Metadata round-trip + legacy back-compat
# --------------------------------------------------------------------------- #

def _build_index(cases: list, embeddings: list) -> EmbeddingIndex:
    index = EmbeddingIndex()
    index.build_index([_unit(e) for e in embeddings], [c.to_metadata() for c in cases])
    return index


def test_metadata_roundtrip_save_load(tmp_path: Path) -> None:
    cases = [
        ReferenceCase("a.png", {"Fracture": 1}, {"view": "PA", "age": 40}),
        ReferenceCase("b.png", {"Effusion": 1}, {"view": "AP", "age": 70}),
    ]
    index = _build_index(cases, [[1, 0], [0, 1]])
    index.save(tmp_path / "idx")

    reloaded = EmbeddingIndex()
    reloaded.load(tmp_path / "idx")
    hits = reloaded.search(_unit([1, 0]), top_k=2)
    top_meta = hits[0][1]
    assert top_meta["path"] == "a.png"
    assert top_meta["labels"] == {"Fracture": 1}
    assert top_meta["attributes"]["view"] == "PA"


def test_load_upgrades_legacy_string_metadata(tmp_path: Path) -> None:
    """An old index whose metadata.json is a bare path list still loads."""
    index = _build_index([ReferenceCase("a.png")], [[1, 0]])
    index.save(tmp_path / "idx")
    # Simulate a legacy cache: overwrite metadata with bare strings.
    (tmp_path / "idx" / "metadata.json").write_text(json.dumps(["a.png"]))

    reloaded = EmbeddingIndex()
    reloaded.load(tmp_path / "idx")
    score, meta = reloaded.search(_unit([1, 0]), top_k=1)[0]
    assert meta == {"path": "a.png", "labels": {}, "attributes": {}}


# --------------------------------------------------------------------------- #
# Attribute / label filtering + hybrid ranking
# --------------------------------------------------------------------------- #

@pytest.fixture
def filter_index() -> EmbeddingIndex:
    cases = [
        ReferenceCase("A.png", {"Fracture": 1}, {"view": "PA", "age": 40}),
        ReferenceCase("B.png", {"Fracture": 1}, {"view": "AP", "age": 70}),
        ReferenceCase("C.png", {"Effusion": 1}, {"view": "PA", "age": 40}),
        ReferenceCase("D.png", {"Fracture": 0}, {}),  # legacy: no attributes
    ]
    embeddings = [[1, 0], [0.8, 0.6], [0, 1], [0.9, 0.1]]
    return _build_index(cases, embeddings)


def _paths(hits) -> list:
    return [m["path"] for _s, m in hits]


def test_required_labels_filters_and_ranks_by_similarity(filter_index) -> None:
    hits = filter_index.search(_unit([1, 0]), top_k=5, filters={"required_labels": ["Fracture"]})
    # Only A and B are Fracture-positive; A is the closer vector → ranked first.
    assert _paths(hits) == ["A.png", "B.png"]


def test_view_filter_excludes_attributeless_cases(filter_index) -> None:
    hits = filter_index.search(
        _unit([1, 0]), top_k=5, filters={"required_labels": ["Fracture"], "view": "PA"})
    assert _paths(hits) == ["A.png"]            # B is AP, D has no view attribute


def test_age_range_filter(filter_index) -> None:
    hits = filter_index.search(
        _unit([1, 0]), top_k=5, filters={"required_labels": ["Fracture"], "age_range": (60, 80)})
    assert _paths(hits) == ["B.png"]


def test_filter_with_no_matches_returns_empty(filter_index) -> None:
    assert filter_index.search(_unit([1, 0]), top_k=5, filters={"required_labels": ["Cardiomegaly"]}) == []


def test_retrieve_reference_cases_returns_matches(filter_index) -> None:
    matches = retrieve_reference_cases(
        _unit([1, 0]), filter_index, filters={"required_labels": ["Fracture"]}, top_k=5)
    assert [m.case.path for m in matches] == ["A.png", "B.png"]
    assert matches[0].similarity == pytest.approx(1.0, abs=1e-5)
    assert isinstance(matches[0].case, ReferenceCase)


# --------------------------------------------------------------------------- #
# Persistent reference index (build-or-load caching)
# --------------------------------------------------------------------------- #

def _make_dataset(folder: Path, items: dict) -> None:
    """Create a dataset folder of empty PNGs + JSON sidecars.

    *items* maps ``stem -> sidecar dict`` (labels + optional ``_attributes``).
    """
    folder.mkdir(parents=True, exist_ok=True)
    for stem, sidecar in items.items():
        (folder / f"{stem}.png").write_bytes(b"")
        (folder / f"{stem}.json").write_text(json.dumps(sidecar))


def test_get_reference_index_builds_then_loads_from_cache(tmp_path: Path, monkeypatch) -> None:
    import src.imaging.retrieval as retrieval

    cache_dir = tmp_path / "refidx"
    monkeypatch.setattr(retrieval, "_reference_cache_dir", lambda: cache_dir)

    folder = tmp_path / "book1"
    _make_dataset(folder, {
        "x0": {"Fracture": 1, "_attributes": {"view": "PA"}},
        "x1": {"Effusion": 1},
    })
    registry = DatasetRegistry(registry_path=tmp_path / "datasets.json")
    registry.register_dataset("book1", str(folder))

    extractor = _FakeExtractor({"x0.png": [1, 0], "x1.png": [0, 1]})
    index = get_reference_index(registry=registry, extractor=extractor)
    assert index is not None
    assert len(extractor.calls) == 2                  # both images embedded once
    assert (cache_dir / "manifest_hash.txt").is_file()

    # Second call: hash unchanged → must load from cache, never extracting again.
    index2 = get_reference_index(registry=registry, extractor=_RaisingExtractor())
    assert index2 is not None
    hits = index2.search(_unit([1, 0]), top_k=1, filters={"required_labels": ["Fracture"]})
    assert _paths(hits) == [str(folder / "x0.png")]


def test_get_reference_index_none_when_no_datasets(tmp_path: Path) -> None:
    registry = DatasetRegistry(registry_path=tmp_path / "datasets.json")
    assert get_reference_index(registry=registry, extractor=_RaisingExtractor()) is None
