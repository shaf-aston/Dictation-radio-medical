"""Tests for SyncManager artifact extraction — the untrusted-archive gate.

The critical regression these guard against: a model archive is downloaded from
the cloud and is therefore untrusted. ``_extract_model`` must reject any member
that would escape the destination directory (CVE-2007-4559 path traversal)
before extracting, and extract benign archives unchanged.
"""

from __future__ import annotations

import io
import tarfile

import pytest

from src.cloud.exceptions import CloudError
from src.cloud.framework.sync import SyncManager


def _make_targz(path, members: dict) -> None:
    """Write a .tar.gz whose members are exactly ``{arcname: bytes}``."""
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_extract_clean_archive(tmp_path):
    archive = tmp_path / "model.tar.gz"
    _make_targz(archive, {"model.bin": b"weights", "config.json": b"{}"})
    dest = tmp_path / "out"

    SyncManager._extract_model(archive, dest)

    assert (dest / "model.bin").read_bytes() == b"weights"
    assert (dest / "config.json").read_bytes() == b"{}"
    # The archive is removed after a successful extract.
    assert not archive.exists()


def test_extract_rejects_path_traversal(tmp_path):
    archive = tmp_path / "evil.tar.gz"
    _make_targz(archive, {"../escaped.txt": b"pwned"})
    dest = tmp_path / "out"

    with pytest.raises(CloudError):
        SyncManager._extract_model(archive, dest)

    # Nothing was written outside the destination directory.
    assert not (tmp_path / "escaped.txt").exists()


def test_extract_rejects_traversal_among_safe_members(tmp_path):
    # A single bad member taints the whole archive — fail closed, write nothing.
    archive = tmp_path / "mixed.tar.gz"
    _make_targz(archive, {"good.bin": b"ok", "../../etc_evil": b"pwned"})
    dest = tmp_path / "out"

    with pytest.raises(CloudError):
        SyncManager._extract_model(archive, dest)

    # Fail closed: the benign member is not extracted either.
    assert not (dest / "good.bin").exists()


# ---------------------------------------------------------------------------
# Task dispatch: a batch trains the model type it was created for.
# ---------------------------------------------------------------------------

def test_start_batch_dispatches_to_task(monkeypatch):
    from datetime import datetime, timezone

    from src.training.schemas import (
        BATCH_TRAINING, TASK_TEXT_CORRECTOR, CorrectionRecord,
    )

    records = [
        CorrectionRecord(
            session_id="s", created_at=datetime.now(timezone.utc).isoformat(),
            wrong_text=f"w{i}", correct_text=f"c{i}", model_version="base",
            accent_profile="neutral", deidentified=True, id=i,
        )
        for i in range(3)
    ]

    submitted = {}

    class _FakeClient:
        def upload_training_data(self, archive):
            return "url://uploaded"

        def submit_training_job(self, spec):
            submitted["entrypoint"] = spec.entrypoint
            submitted["name"] = spec.name
            return "job-123"

    class _FakeDB:
        def __init__(self):
            self.batch = None
            self.status = None

        def get_pending(self, limit=None):
            return records

        def insert_batch(self, batch):
            self.batch = batch

        def mark_uploaded(self, ids, batch_id):
            pass

        def update_batch_status(self, batch_id, status, lightning_job_id=None):
            self.status = (status, lightning_job_id)

    sync = SyncManager.__new__(SyncManager)   # bypass real DB init
    sync._db = _FakeDB()  # type: ignore
    monkeypatch.setattr("src.features.audit_log.log_training_upload",
                        lambda *a, **k: None)

    batch_id = sync._start_batch(
        _FakeClient(), TASK_TEXT_CORRECTOR, base_model="t5-small", min_records=2)

    assert batch_id is not None and batch_id.startswith(TASK_TEXT_CORRECTOR)
    assert submitted["entrypoint"] == "scripts/lightning/train_text_corrector.py"
    assert sync._db.batch is not None and sync._db.batch.task_type == TASK_TEXT_CORRECTOR  # type: ignore
    assert sync._db.status == (BATCH_TRAINING, "job-123")  # type: ignore
