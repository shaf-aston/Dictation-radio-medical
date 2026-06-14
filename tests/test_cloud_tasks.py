"""Tests for the cloud-training task plug-ins (archive builders + job specs).

These run without any ML dependency: they only exercise the framework seam each
task implements — how it bundles data and how it describes its Lightning job.
"""

from __future__ import annotations

import json
import tarfile
from datetime import datetime, timezone

import pytest

from src.cloud.exceptions import CloudError
from src.cloud.tasks import TASKS
from src.cloud.tasks.scan_finetune import ScanClassifierTask
from src.cloud.tasks.text_corrector import TextCorrectorTask
from src.training.schemas import (
    TASK_SCAN_CLASSIFIER,
    TASK_TEXT_CORRECTOR,
    TASK_WHISPER_VOICE,
    CorrectionRecord,
)


def _record(wrong, correct) -> CorrectionRecord:
    return CorrectionRecord(
        session_id="s", created_at=datetime.now(timezone.utc).isoformat(),
        wrong_text=wrong, correct_text=correct, model_version="base",
        accent_profile="neutral", deidentified=True,
    )


def _read_manifest(archive):
    with tarfile.open(archive, "r:gz") as tar:
        return json.loads(
            tar.extractfile("manifest.json").read().decode("utf-8"))  # type: ignore[union-attr]


def test_registry_has_all_three_tasks():
    assert set(TASKS) == {
        TASK_WHISPER_VOICE, TASK_TEXT_CORRECTOR, TASK_SCAN_CLASSIFIER}


# ---------------------------------------------------------------------------
# Text corrector
# ---------------------------------------------------------------------------

def test_text_corrector_archive_filters_noop_pairs():
    records = [
        _record("akl", "ACL"),
        _record("same", "same"),     # no-op → excluded
        _record("", "x"),            # empty wrong → excluded
    ]
    archive = TextCorrectorTask().build_archive("b1", records, base_model="t5-small")
    try:
        manifest = _read_manifest(archive)
        assert manifest["task_type"] == TASK_TEXT_CORRECTOR
        assert manifest["pairs"] == [{"wrong": "akl", "correct": "ACL"}]
    finally:
        archive.unlink(missing_ok=True)


def test_text_corrector_job_spec():
    spec = TextCorrectorTask().job_spec("b1", "url://x", "t5-small")
    assert spec.entrypoint == "scripts/lightning/train_text_corrector.py"
    assert spec.args["data-url"] == "url://x"


# ---------------------------------------------------------------------------
# Scan classifier
# ---------------------------------------------------------------------------

def test_scan_archive_bundles_labelled_images(tmp_path, monkeypatch):
    import src.features.file_manager as fm
    train_dir = tmp_path / "imaging" / "training"
    train_dir.mkdir(parents=True)
    (train_dir / "a.png").write_bytes(b"PNGDATA")
    (train_dir / "a.json").write_text(json.dumps({"Cardiomegaly": 1}), encoding="utf-8")
    (train_dir / "orphan.png").write_bytes(b"PNG")  # no label → skipped
    monkeypatch.setattr(fm, "imaging_training_dir", lambda: train_dir)

    archive = ScanClassifierTask().build_archive("b1", [], base_model="dn121")
    try:
        with tarfile.open(archive, "r:gz") as tar:
            names = tar.getnames()
        manifest = _read_manifest(archive)
        assert [e["image_file"] for e in manifest["examples"]] == ["images/a.png"]
        assert "images/a.png" in names
        assert "images/orphan.png" not in names
    finally:
        archive.unlink(missing_ok=True)


def test_scan_archive_oversamples_trauma_labels(tmp_path, monkeypatch):
    import src.features.file_manager as fm
    train_dir = tmp_path / "imaging" / "training"
    train_dir.mkdir(parents=True)
    (train_dir / "trauma.png").write_bytes(b"PNGDATA")
    (train_dir / "trauma.json").write_text(
        json.dumps({"Fracture": 1, "Cardiomegaly": 0}), encoding="utf-8")
    (train_dir / "routine.png").write_bytes(b"PNGDATA")
    (train_dir / "routine.json").write_text(
        json.dumps({"Cardiomegaly": 1}), encoding="utf-8")
    monkeypatch.setattr(fm, "imaging_training_dir", lambda: train_dir)

    archive = ScanClassifierTask().build_archive("b1", [], base_model="dn121")
    try:
        with tarfile.open(archive, "r:gz") as tar:
            names = tar.getnames()
        manifest = _read_manifest(archive)
        files = [e["image_file"] for e in manifest["examples"]]

        # Trauma-positive example is oversampled in the manifest...
        assert files.count("images/trauma.png") == 3
        assert files.count("images/routine.png") == 1
        # ...but each underlying image is archived only once.
        assert names.count("images/trauma.png") == 1
        assert names.count("images/routine.png") == 1
        assert manifest["trauma_focus"] == {
            "labels": ["Effusion", "Fracture", "Pneumothorax"],
            "oversample_factor": 3,
        }
    finally:
        archive.unlink(missing_ok=True)


def test_scan_archive_raises_without_labels(tmp_path, monkeypatch):
    import src.features.file_manager as fm
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(fm, "imaging_training_dir", lambda: empty)
    with pytest.raises(CloudError):
        ScanClassifierTask().build_archive("b1", [], base_model="dn121")
