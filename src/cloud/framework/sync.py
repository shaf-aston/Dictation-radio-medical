"""Orchestrates the full upload → train → download → register cycle.

This is the high-level façade the UI talks to; it hides the client, tasks, DB,
and registry behind three intentions:

  * ``maybe_start_training`` — if enough data has accrued for a task, kick off a job.
  * ``poll_and_collect`` — advance any in-flight jobs; download + register
    finished models (for any task).
  * status helpers for the settings panel.

Which model type a batch trains is recorded on the batch (``task_type``) and
dispatched through :data:`src.cloud.tasks.TASKS`, so the same code path drives
voice, text-correction, and scan models. All methods are safe to call when cloud
is disabled (they return early), and all network errors are caught and logged
rather than propagated to the UI thread.
"""

from __future__ import annotations

import logging
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple
from uuid import uuid4

from src.cloud.exceptions import CloudError
from src.training.schemas import (
    BATCH_DONE,
    BATCH_FAILED,
    BATCH_TRAINING,
    BATCH_UPLOADING,
    TASK_WHISPER_VOICE,
    TrainingBatch,
)

logger = logging.getLogger(__name__)


class SyncManager:
    """Coordinates cloud training end-to-end across all task types."""

    def __init__(self) -> None:
        from src.training.staging_db import get_staging_db
        self._db = get_staging_db()

    # ------------------------------------------------------------------
    # Settings / client helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _enabled() -> bool:
        try:
            from src.core.settings import Settings
            s = Settings()
            return bool(s.get("cloud_enabled")) and bool(s.get("cloud_training_consent"))
        except Exception as exc:
            logger.debug("Could not read cloud settings; treating as disabled: %s", exc)
            return False

    def _make_client(self):
        from src.core.settings import Settings
        from src.cloud.framework.client import LightningAIClient
        if project_id := Settings().get("lightning_project_id") or "":
            return LightningAIClient(project_id=project_id)
        raise CloudError("No Lightning AI project id configured.")

    # ------------------------------------------------------------------
    # Start training
    # ------------------------------------------------------------------

    def maybe_start_training(
        self, task_type: str = TASK_WHISPER_VOICE
    ) -> Optional[str]:
        """Upload + submit a job for *task_type* if its threshold is met.

        Returns the new ``batch_id`` if a job was submitted, else None.
        """
        if not self._enabled():
            return None
        from src.core.settings import Settings
        s = Settings()
        min_records = int(s.get("min_corrections_before_upload", 20))
        if self._db.pending_count() < min_records:
            return None
        try:
            client = self._make_client()
            return self._start_batch(
                client, task_type,
                base_model=s.get("model_size", "base"),
                min_records=min_records,
            )
        except CloudError as exc:
            logger.warning("Could not start %s training: %s", task_type, exc)
            return None

    def _start_batch(
        self, client, task_type: str, base_model: str,
        min_records: int, max_records: int = 500,
    ) -> Optional[str]:
        """Bundle pending records via the task, upload, and submit a job."""
        from src.cloud.tasks import TASKS
        from src.features import audit_log

        task = TASKS[task_type]
        records = self._db.get_pending(limit=max_records)
        if len(records) < min_records:
            logger.info("Only %d pending records (<%d); skipping %s upload",
                        len(records), min_records, task_type)
            return None

        batch_id = (f"{task_type}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
                    f"_{uuid4().hex[:6]}")
        archive = task.build_archive(batch_id, records, base_model)

        self._db.insert_batch(TrainingBatch(
            batch_id=batch_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            record_count=len(records),
            status=BATCH_UPLOADING,
            task_type=task_type,
        ))
        audit_log.log_training_upload(batch_id, len(records))

        try:
            data_url = client.upload_training_data(archive)
            job_id = client.submit_training_job(
                task.job_spec(batch_id, data_url, base_model)
            )
        except CloudError:
            self._db.update_batch_status(batch_id, BATCH_FAILED)
            raise
        finally:
            try:
                archive.unlink()
            except OSError:
                pass

        self._db.mark_uploaded([r.id for r in records if r.id is not None], batch_id)
        self._db.update_batch_status(batch_id, BATCH_TRAINING, lightning_job_id=job_id)
        logger.info("Batch %s uploaded; %s job %s submitted", batch_id, task_type, job_id)
        return batch_id

    # ------------------------------------------------------------------
    # Poll in-flight jobs
    # ------------------------------------------------------------------

    def poll_and_collect(self) -> List[str]:
        """Check active jobs (all tasks); download + register any that finished.

        Returns the list of newly-registered model versions.
        """
        if not self._enabled():
            return []
        active = self._db.get_active_batches()
        if not active:
            return []
        try:
            client = self._make_client()
        except CloudError as exc:
            logger.warning("Cannot poll jobs: %s", exc)
            return []

        new_versions: List[str] = []
        for batch in active:
            if not batch.lightning_job_id:
                continue
            try:
                status = client.get_job_status(batch.lightning_job_id)
            except CloudError as exc:
                logger.warning("Status check failed for %s: %s", batch.batch_id, exc)
                continue
            if status["status"] == "completed":
                if version := self._download_and_register(client, batch, status):
                    new_versions.append(version)
            elif status["status"] == "failed":
                self._db.update_batch_status(batch.batch_id, BATCH_FAILED)
                logger.warning("Training job for %s failed: %s",
                               batch.batch_id, status.get("error"))
        return new_versions

    def _download_and_register(self, client, batch, status) -> Optional[str]:
        from src.cloud.framework.registry import ModelRegistry
        from src.core.settings import Settings
        from src.features.file_manager import fine_tuned_dir

        artifact_url = status.get("artifact_url")
        if not artifact_url:
            logger.warning("Completed job %s has no artifact url", batch.batch_id)
            self._db.update_batch_status(batch.batch_id, BATCH_FAILED)
            return None

        version = batch.batch_id
        dest = fine_tuned_dir() / version
        try:
            archive = client.download_artifact(artifact_url, dest)
            self._extract_model(archive, dest)
        except (CloudError, OSError) as exc:
            logger.warning("Could not retrieve model for %s: %s", batch.batch_id, exc)
            self._db.update_batch_status(batch.batch_id, BATCH_FAILED)
            return None

        ModelRegistry().register_downloaded_model(
            version=version,
            local_path=dest,
            base_model=Settings().get("model_size", "base"),
            task_type=batch.task_type,
            correction_count=batch.record_count,
            lightning_job_id=batch.lightning_job_id,
        )
        self._db.update_batch_status(batch.batch_id, BATCH_DONE)
        logger.info("Model %s (%s) downloaded and registered",
                    version, batch.task_type)
        return version

    @staticmethod
    def _extract_model(archive: Path, dest: Path) -> None:
        """Extract a downloaded model archive into *dest*, rejecting unsafe paths.

        The archive comes from the cloud, so it is untrusted. Every member is
        validated to resolve *inside* ``dest`` before extraction — otherwise a
        crafted ``../`` or absolute member could overwrite files outside the
        model directory (CVE-2007-4559). ``filter="data"`` adds the standard tar
        hardening on Python 3.12+.
        """
        dest = Path(dest).resolve()
        with tarfile.open(archive, "r:gz") as tar:
            for m in tar.getmembers():
                target = (dest / m.name).resolve()
                if target != dest and dest not in target.parents:
                    raise CloudError(f"Unsafe path in artifact: {m.name}")
            tar.extractall(dest, filter="data")
        try:
            archive.unlink()
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Status for UI
    # ------------------------------------------------------------------

    def training_progress(self) -> Tuple[int, int]:
        """Return (pending_corrections, threshold) for a progress indicator."""
        from src.core.settings import Settings
        threshold = int(Settings().get("min_corrections_before_upload", 20))
        return self._db.pending_count(), threshold
