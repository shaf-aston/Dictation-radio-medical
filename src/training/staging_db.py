"""SQLite staging store for cloud training data.

Holds correction records, training batches, and a mirror of the model-version
registry. SQLite is used (rather than JSON) because the correction table is
queried by status and grouped into batches — relational access the existing
flat-JSON stores don't serve well. The DB lives at ``data/training/staging.db``
and is created lazily on first use.

Thread-safe: a single module-level connection is guarded by a lock, and the
connection is opened with ``check_same_thread=False`` so the UI thread and the
worker thread can both record corrections.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import List, Optional

from src.training.schemas import (
    BATCH_PENDING,
    CorrectionRecord,
    TrainingBatch,
    UPLOAD_PENDING,
)

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS correction_records (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    wrong_text    TEXT NOT NULL,
    correct_text  TEXT NOT NULL,
    audio_path    TEXT,
    ts_start      REAL,
    ts_end        REAL,
    model_version TEXT NOT NULL,
    accent_profile TEXT NOT NULL,
    upload_status TEXT NOT NULL DEFAULT 'pending',
    deidentified  INTEGER NOT NULL DEFAULT 0,
    batch_id      TEXT
);

CREATE TABLE IF NOT EXISTS training_batches (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id         TEXT UNIQUE NOT NULL,
    created_at       TEXT NOT NULL,
    record_count     INTEGER NOT NULL,
    upload_at        TEXT,
    lightning_job_id TEXT,
    status           TEXT NOT NULL DEFAULT 'pending'
);

CREATE INDEX IF NOT EXISTS idx_corr_status ON correction_records(upload_status);
CREATE INDEX IF NOT EXISTS idx_corr_batch ON correction_records(batch_id);
"""


class StagingDB:
    """Thin SQLite wrapper for the training staging store."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        if db_path is None:
            from src.features.file_manager import staging_db_path
            db_path = staging_db_path()
        self._path = Path(db_path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        logger.info("StagingDB ready at %s", self._path)

    # ------------------------------------------------------------------
    # Correction records
    # ------------------------------------------------------------------

    def insert_correction(self, record: CorrectionRecord) -> int:
        """Insert a correction record, returning its new row id."""
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO correction_records
                   (session_id, created_at, wrong_text, correct_text, audio_path,
                    ts_start, ts_end, model_version, accent_profile,
                    upload_status, deidentified)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.session_id, record.created_at, record.wrong_text,
                    record.correct_text, record.audio_path, record.ts_start,
                    record.ts_end, record.model_version, record.accent_profile,
                    record.upload_status, int(record.deidentified),
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def pending_count(self) -> int:
        """Number of records ready to upload (deidentified and not yet sent)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM correction_records "
                "WHERE upload_status = ? AND deidentified = 1",
                (UPLOAD_PENDING,),
            ).fetchone()
        return int(row["n"])

    def get_pending(self, limit: Optional[int] = None) -> List[CorrectionRecord]:
        """Fetch de-identified, not-yet-uploaded records oldest-first."""
        sql = (
            "SELECT * FROM correction_records "
            "WHERE upload_status = ? AND deidentified = 1 ORDER BY id ASC"
        )
        params: tuple = (UPLOAD_PENDING,)
        if limit is not None:
            sql += " LIMIT ?"
            params = (UPLOAD_PENDING, limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_record(r) for r in rows]

    def mark_uploaded(self, record_ids: List[int], batch_id: str) -> None:
        if not record_ids:
            return
        placeholders = ",".join("?" * len(record_ids))
        with self._lock:
            self._conn.execute(
                f"UPDATE correction_records SET upload_status = 'uploaded', "
                f"batch_id = ? WHERE id IN ({placeholders})",
                (batch_id, *record_ids),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Training batches
    # ------------------------------------------------------------------

    def insert_batch(self, batch: TrainingBatch) -> int:
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO training_batches
                   (batch_id, created_at, record_count, status, lightning_job_id)
                   VALUES (?, ?, ?, ?, ?)""",
                (batch.batch_id, batch.created_at, batch.record_count,
                 batch.status, batch.lightning_job_id),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def update_batch_status(
        self, batch_id: str, status: str, lightning_job_id: Optional[str] = None
    ) -> None:
        with self._lock:
            if lightning_job_id is not None:
                self._conn.execute(
                    "UPDATE training_batches SET status = ?, lightning_job_id = ? "
                    "WHERE batch_id = ?",
                    (status, lightning_job_id, batch_id),
                )
            else:
                self._conn.execute(
                    "UPDATE training_batches SET status = ? WHERE batch_id = ?",
                    (status, batch_id),
                )
            self._conn.commit()

    def get_active_batches(self) -> List[TrainingBatch]:
        """Batches that are uploading or training (need status polling)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM training_batches WHERE status IN ('uploading', 'training')"
            ).fetchall()
        return [self._row_to_batch(r) for r in rows]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_record(r: sqlite3.Row) -> CorrectionRecord:
        return CorrectionRecord(
            id=r["id"],
            session_id=r["session_id"],
            created_at=r["created_at"],
            wrong_text=r["wrong_text"],
            correct_text=r["correct_text"],
            audio_path=r["audio_path"],
            ts_start=r["ts_start"],
            ts_end=r["ts_end"],
            model_version=r["model_version"],
            accent_profile=r["accent_profile"],
            upload_status=r["upload_status"],
            deidentified=bool(r["deidentified"]),
        )

    @staticmethod
    def _row_to_batch(r: sqlite3.Row) -> TrainingBatch:
        return TrainingBatch(
            id=r["id"],
            batch_id=r["batch_id"],
            created_at=r["created_at"],
            record_count=r["record_count"],
            status=r["status"],
            upload_at=r["upload_at"],
            lightning_job_id=r["lightning_job_id"],
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_staging_db: Optional[StagingDB] = None
_singleton_lock = threading.Lock()


def get_staging_db() -> StagingDB:
    """Return the process-wide StagingDB instance (created on first call)."""
    global _staging_db
    if _staging_db is None:
        with _singleton_lock:
            if _staging_db is None:
                _staging_db = StagingDB()
    return _staging_db
