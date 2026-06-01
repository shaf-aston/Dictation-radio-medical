"""Tests for the SQLite training-data staging store."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.training.schemas import (
    BATCH_TRAINING,
    CorrectionRecord,
    TrainingBatch,
)
from src.training.staging_db import StagingDB


def _record(wrong="wertebra", correct="vertebra", deidentified=True) -> CorrectionRecord:
    return CorrectionRecord(
        session_id="sess1",
        created_at=datetime.now(timezone.utc).isoformat(),
        wrong_text=wrong,
        correct_text=correct,
        model_version="base",
        accent_profile="south_asian",
        deidentified=deidentified,
    )


@pytest.fixture
def db(tmp_path):
    return StagingDB(db_path=tmp_path / "staging.db")


def test_insert_and_pending_count(db):
    db.insert_correction(_record())
    db.insert_correction(_record(wrong="tier", correct="tear"))
    assert db.pending_count() == 2


def test_non_deidentified_excluded_from_pending(db):
    db.insert_correction(_record(deidentified=False))
    assert db.pending_count() == 0
    assert db.get_pending() == []


def test_mark_uploaded_removes_from_pending(db):
    rid = db.insert_correction(_record())
    db.mark_uploaded([rid], "batch_x")
    assert db.pending_count() == 0


def test_get_pending_respects_limit(db):
    for _ in range(5):
        db.insert_correction(_record())
    assert len(db.get_pending(limit=3)) == 3


def test_batch_lifecycle(db):
    db.insert_batch(TrainingBatch(
        batch_id="b1",
        created_at=datetime.now(timezone.utc).isoformat(),
        record_count=10,
        status="uploading",
    ))
    db.update_batch_status("b1", BATCH_TRAINING, lightning_job_id="job-42")
    active = db.get_active_batches()
    assert len(active) == 1
    assert active[0].lightning_job_id == "job-42"
    assert active[0].status == BATCH_TRAINING
