"""Orchestrates the full upload → train → download → register cycle.

This is the high-level façade the UI talks to; it hides the client, uploader,
DB, and registry behind three intentions:

  * ``maybe_start_training`` — if enough corrections have accrued, kick off a job.
  * ``poll_and_collect`` — advance any in-flight jobs; download + register
    finished models.
  * status helpers for the settings panel.

All methods are safe to call when cloud is disabled (they return early), and all
network errors are caught and logged rather than propagated to the UI thread.
"""

from __future__ import annotations

import logging
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from src.cloud.exceptions import CloudError
from src.training.schemas import BATCH_DONE, BATCH_FAILED

logger = logging.getLogger(__name__)


class SyncManager:
    """Coordinates cloud training end-to-end."""

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
        except Exception:
            return False

    def _make_client(self):
        from src.core.settings import Settings
        from src.cloud.client import LightningAIClient
        project_id = Settings().get("lightning_project_id") or ""
        if not project_id:
            raise CloudError("No Lightning AI project id configured.")
        return LightningAIClient(project_id=project_id)

    # ------------------------------------------------------------------
    # Start training
    # ------------------------------------------------------------------

    def maybe_start_training(self) -> Optional[str]:
        """Upload + submit a job if the pending-correction threshold is met."""
        if not self._enabled():
            return None
        from src.core.settings import Settings
        from src.cloud.uploader import DataUploader
        s = Settings()
        min_records = int(s.get("min_corrections_before_upload", 20))
        if self._db.pending_count() < min_records:
            return None
        try:
            client = self._make_client()
            uploader = DataUploader(client, self._db)
            return uploader.create_and_upload_batch(
                base_model=s.get("model_size", "base"), min_records=min_records
            )
        except CloudError as exc:
            logger.warning("Could not start training: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Poll in-flight jobs
    # ------------------------------------------------------------------

    def poll_and_collect(self) -> List[str]:
        """Check active jobs; download + register any that finished.

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
                version = self._download_and_register(client, batch, status)
                if version:
                    new_versions.append(version)
            elif status["status"] == "failed":
                self._db.update_batch_status(batch.batch_id, BATCH_FAILED)
                logger.warning("Training job for %s failed: %s",
                               batch.batch_id, status.get("error"))
        return new_versions

    def _download_and_register(self, client, batch, status) -> Optional[str]:
        from src.cloud.model_registry import ModelRegistry
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
            correction_count=batch.record_count,
            lightning_job_id=batch.lightning_job_id,
        )
        self._db.update_batch_status(batch.batch_id, BATCH_DONE)
        logger.info("Model %s downloaded and registered", version)
        return version

    @staticmethod
    def _extract_model(archive: Path, dest: Path) -> None:
        """Extract the CT2 model archive into *dest* (flattening one level)."""
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(dest)
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
