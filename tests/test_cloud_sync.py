"""A finished training job is collected once: a bad model never loops."""
import io
import tarfile

import pytest

from src.cloud.framework import sync
from src.core import paths
from src.training.schemas import BATCH_DONE, BATCH_FAILED, BATCH_TRAINING, TrainingBatch
from src.training.staging_db import StagingDB


def _good_archive() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = b"weights"
        info = tarfile.TarInfo("model.bin")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class Client:
    def __init__(self, payloads):
        self.payloads = payloads
        self.downloads = 0

    def get_job_status(self, job_id):
        return {"status": "completed", "artifact_url": f"https://lightning.ai/{job_id}"}

    def download_artifact(self, url, dest):
        self.downloads += 1
        dest.mkdir(parents=True, exist_ok=True)
        archive = dest / "model_artifact.tar.gz"
        archive.write_bytes(self.payloads[url.rsplit("/", 1)[1]])
        return archive


@pytest.fixture
def manager(tmp_path, monkeypatch):
    paths._data_dir.cache_clear()
    monkeypatch.setattr(paths, "_data_dir", lambda: tmp_path)
    db = StagingDB(tmp_path / "staging.db")
    for job in ("bad", "good"):
        db.insert_batch(TrainingBatch(batch_id=f"b_{job}", created_at="t", record_count=1,
                                      status=BATCH_TRAINING, lightning_job_id=job))
    client = Client({"bad": b"not a tar file", "good": _good_archive()})
    m = sync.SyncManager.__new__(sync.SyncManager)
    m._db = db
    monkeypatch.setattr(sync.SyncManager, "_enabled", staticmethod(lambda: True))
    monkeypatch.setattr(sync.SyncManager, "_make_client", lambda self: client)
    return m, db, client


def _status(db, batch_id):
    row = db._conn.execute("SELECT status FROM training_batches WHERE batch_id = ?",
                           (batch_id,)).fetchone()
    return row[0]


def test_a_corrupt_model_fails_its_batch_and_leaves_nothing_behind(manager):
    m, db, client = manager
    assert m.poll_and_collect() == ["b_good"]
    assert _status(db, "b_bad") == BATCH_FAILED
    assert _status(db, "b_good") == BATCH_DONE
    assert not (paths.fine_tuned_dir() / "b_bad").exists()
    assert (paths.fine_tuned_dir() / "b_good" / "model.bin").exists()
    # Nothing is fetched again on the next poll.
    assert m.poll_and_collect() == [] and client.downloads == 2


def test_scan_uploads_carry_no_filename_or_local_account(tmp_path, monkeypatch):
    import json as _json

    from src.cloud.tasks.scan_finetune import ScanClassifierTask
    paths._data_dir.cache_clear()
    monkeypatch.setattr(paths, "_data_dir", lambda: tmp_path)
    scans = paths.imaging_training_dir()
    for name, data in (("Jane_Doe_MRN12345678.png", b"img-a"), ("0001.png", b"img-b")):
        (scans / name).write_bytes(data)
        (scans / name).with_suffix(".json").write_text(_json.dumps({"Fracture": 1}))
    archive = ScanClassifierTask().build_archive("b1", [], "")
    try:
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            manifest = _json.load(tar.extractfile("manifest.json"))
    finally:
        archive.unlink()
    names = [m.name for m in members]
    assert len(names) == 3 and not any("Jane" in n or "0001" in n for n in names)
    assert {e["image_file"] for e in manifest["examples"]} <= set(names)
    assert all(m.uname == "" and m.uid == 0 for m in members)
