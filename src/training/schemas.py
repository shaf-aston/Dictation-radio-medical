"""Dataclasses shared across the training/cloud subsystem.

These mirror the SQLite tables in :mod:`src.training.staging_db` and the
``registry.json`` index in :mod:`src.cloud.framework.registry`. Keeping them as
plain dataclasses (no ORM) matches the project's lightweight, file-first style.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

# Upload lifecycle for a correction record.
UPLOAD_PENDING = "pending"
UPLOAD_UPLOADED = "uploaded"
UPLOAD_FAILED = "failed"

# Lifecycle for a training batch / cloud job.
BATCH_PENDING = "pending"
BATCH_UPLOADING = "uploading"
BATCH_TRAINING = "training"
BATCH_DONE = "done"
BATCH_FAILED = "failed"

# Cloud-training task types. The registry and batches are keyed by these so one
# Lightning account can hold voice, text-correction, and scan models side by side.
TASK_WHISPER_VOICE = "whisper_voice"
TASK_TEXT_CORRECTOR = "text_corrector"
TASK_SCAN_CLASSIFIER = "scan_classifier"


@dataclass
class CorrectionRecord:
    """A single (audio, wrong, correct) training example.

    ``audio_path`` points at a de-identified WAV clip on local disk; it is
    nullable because text-only corrections (manual edits with no retained
    audio) are still useful for language-side adaptation.
    """

    session_id: str
    created_at: str               # ISO8601 UTC
    wrong_text: str
    correct_text: str
    model_version: str            # base/fine-tuned version that produced wrong_text
    accent_profile: str
    audio_path: Optional[str] = None
    ts_start: Optional[float] = None
    ts_end: Optional[float] = None
    upload_status: str = UPLOAD_PENDING
    deidentified: bool = False
    id: Optional[int] = None      # set by the DB on insert


@dataclass
class ImageLabelRecord:
    """A single labeled chest X-ray image for training.

    User provides the image path and confidence labels for each pathology.
    The system de-identifies the image before staging for upload.
    """

    image_path: str
    labels: Dict[str, float]      # pathology → confidence (0–1)
    timestamp: str                # ISO8601 UTC
    consent_flags: Dict[str, bool]
    de_identified_image_path: Optional[str] = None
    id: Optional[int] = None      # set by the DB on insert


@dataclass
class TrainingBatch:
    """A group of correction records bundled for one cloud training run."""

    batch_id: str
    created_at: str
    record_count: int
    status: str = BATCH_PENDING
    task_type: str = TASK_WHISPER_VOICE
    upload_at: Optional[str] = None
    lightning_job_id: Optional[str] = None
    id: Optional[int] = None


@dataclass
class ModelVersion:
    """A fine-tuned model version tracked in the registry."""

    version: str
    base_model: str
    created_at: str
    task_type: str = TASK_WHISPER_VOICE
    correction_count: int = 0
    lightning_job_id: Optional[str] = None
    downloaded_at: Optional[str] = None
    local_path: Optional[str] = None     # path to model artifact dir, None until downloaded
    is_active: bool = False
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "base_model": self.base_model,
            "created_at": self.created_at,
            "task_type": self.task_type,
            "correction_count": self.correction_count,
            "lightning_job_id": self.lightning_job_id,
            "downloaded_at": self.downloaded_at,
            "local_path": self.local_path,
            "is_active": self.is_active,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelVersion":
        return cls(
            version=data["version"],
            base_model=data.get("base_model", "base"),
            created_at=data.get("created_at", ""),
            task_type=data.get("task_type", TASK_WHISPER_VOICE),
            correction_count=data.get("correction_count", 0),
            lightning_job_id=data.get("lightning_job_id"),
            downloaded_at=data.get("downloaded_at"),
            local_path=data.get("local_path"),
            is_active=bool(data.get("is_active", False)),
            metrics=data.get("metrics", {}),
        )
