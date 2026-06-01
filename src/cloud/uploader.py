"""Bundles staged corrections into a batch archive and uploads it.

A batch is a ``.tar.gz`` containing every referenced (already de-identified)
audio clip plus a ``manifest.json`` describing the correction triples. The
archive format is exactly what ``scripts/lightning/train_whisper.py`` expects on
the Lightning AI side, so the two must evolve together.
"""

from __future__ import annotations

import json
import logging
import tarfile
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from src.cloud.client import LightningAIClient
from src.cloud.exceptions import CloudError
from src.training.schemas import (
    BATCH_TRAINING,
    BATCH_UPLOADING,
    CorrectionRecord,
    TrainingBatch,
)
from src.training.staging_db import StagingDB, get_staging_db

logger = logging.getLogger(__name__)


class DataUploader:
    """Creates and uploads training batches from the staging DB."""

    def __init__(self, client: LightningAIClient, db: Optional[StagingDB] = None) -> None:
        self._client = client
        self._db = db or get_staging_db()

    def create_and_upload_batch(
        self, base_model: str, min_records: int = 20, max_records: int = 500,
        lora_rank: int = 8, epochs: int = 5,
    ) -> Optional[str]:
        """Bundle pending records, upload, and submit a training job.

        Returns the new ``batch_id`` if a job was submitted, or None if there
        were too few pending records to bother training.
        """
        records = self._db.get_pending(limit=max_records)
        if len(records) < min_records:
            logger.info("Only %d pending records (<%d); skipping upload",
                        len(records), min_records)
            return None

        batch_id = f"batch_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
        archive = self._build_archive(batch_id, records, base_model)

        self._db.insert_batch(TrainingBatch(
            batch_id=batch_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            record_count=len(records),
            status=BATCH_UPLOADING,
        ))
        from src.features import audit_log
        audit_log.log_training_upload(batch_id, len(records))

        try:
            data_url = self._client.upload_training_data(archive)
            job_id = self._client.submit_training_job(
                batch_id, data_url, base_model, lora_rank=lora_rank, epochs=epochs
            )
        except CloudError:
            self._db.update_batch_status(batch_id, "failed")
            raise
        finally:
            try:
                archive.unlink()
            except OSError:
                pass

        self._db.mark_uploaded([r.id for r in records if r.id is not None], batch_id)
        self._db.update_batch_status(batch_id, BATCH_TRAINING, lightning_job_id=job_id)
        logger.info("Batch %s uploaded and training job %s submitted", batch_id, job_id)
        return batch_id

    # ------------------------------------------------------------------
    # Archive construction
    # ------------------------------------------------------------------

    def _build_archive(
        self, batch_id: str, records: List[CorrectionRecord], base_model: str
    ) -> Path:
        """Write a tar.gz of audio clips + manifest.json to a temp file."""
        manifest = {
            "batch_id": batch_id,
            "base_model": base_model,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "records": [],
        }
        # Deduplicate audio clips — many records share one session clip.
        audio_members: dict = {}
        for rec in records:
            entry = {
                "wrong": rec.wrong_text,
                "correct": rec.correct_text,
                "accent": rec.accent_profile,
                "ts_start": rec.ts_start,
                "ts_end": rec.ts_end,
                "audio_file": None,
            }
            if rec.audio_path and Path(rec.audio_path).is_file():
                arcname = f"audio/{Path(rec.audio_path).name}"
                audio_members[rec.audio_path] = arcname
                entry["audio_file"] = arcname
            manifest["records"].append(entry)

        tmp = Path(tempfile.gettempdir()) / f"{batch_id}.tar.gz"
        with tarfile.open(tmp, "w:gz") as tar:
            manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
            info = tarfile.TarInfo("manifest.json")
            info.size = len(manifest_bytes)
            import io
            tar.addfile(info, io.BytesIO(manifest_bytes))
            for src_path, arcname in audio_members.items():
                tar.add(src_path, arcname=arcname)
        logger.info("Built batch archive %s (%d records, %d clips)",
                    tmp.name, len(records), len(audio_members))
        return tmp
