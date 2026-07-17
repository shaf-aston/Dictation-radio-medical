"""Registry of downloaded fine-tuned models — task-aware source of truth.

State lives in ``data/models/registry.json`` so it is human-inspectable and
survives without the cloud being reachable. Each task type (voice, text, scan)
has at most one active version at a time; ``get_active_model_path(task_type)``
returns its artifact directory, or None to mean "use the stock base model".
Activation is reversible via ``rollback_to_base(task_type)``.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from src.core.json_store import read_json, write_json
from src.training.schemas import TASK_WHISPER_VOICE, ModelVersion

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Read/write access to the fine-tuned model index, keyed by task type."""

    def __init__(self, registry_path: Optional[Path] = None) -> None:
        if registry_path is None:
            from src.features.file_manager import model_registry_path
            registry_path = model_registry_path()
        self._path = Path(registry_path)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Load / save
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        data = read_json(self._path, {"active": {}, "models": []})
        return self._normalize(data)

    @staticmethod
    def _normalize(data: dict) -> dict:
        """Upgrade a legacy single-task registry to the task-keyed layout.

        Older files used a flat ``active_version`` string (voice-only). Map it to
        ``active[whisper_voice]`` so existing installs keep their active model.
        """
        if "active" not in data:
            legacy = data.get("active_version")
            data["active"] = {TASK_WHISPER_VOICE: legacy} if legacy else {}
        data.setdefault("models", [])
        return data

    def _save(self, data: dict) -> None:
        write_json(self._path, data)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_downloaded_model(
        self,
        version: str,
        local_path: Path,
        base_model: str,
        task_type: str = TASK_WHISPER_VOICE,
        correction_count: int = 0,
        lightning_job_id: Optional[str] = None,
        metrics: Optional[dict] = None,
    ) -> ModelVersion:
        """Record a freshly-downloaded model. Does not activate it."""
        now = datetime.now(timezone.utc).isoformat()
        mv = ModelVersion(
            version=version,
            base_model=base_model,
            created_at=now,
            task_type=task_type,
            correction_count=correction_count,
            lightning_job_id=lightning_job_id,
            downloaded_at=now,
            local_path=str(local_path),
            is_active=False,
            metrics=metrics or {},
        )
        with self._lock:
            data = self._load()
            # Replace any existing entry with the same version.
            data["models"] = [m for m in data["models"] if m.get("version") != version]
            data["models"].append(mv.to_dict())
            self._save(data)
        logger.info("Registered %s model %s at %s", task_type, version, local_path)
        return mv

    # ------------------------------------------------------------------
    # Activation
    # ------------------------------------------------------------------

    def activate_model(self, version: str) -> bool:
        """Make *version* the active model for its task. False if unknown."""
        with self._lock:
            data = self._load()
            target = next((m for m in data["models"] if m.get("version") == version), None)
            if target is None or not target.get("local_path"):
                logger.warning("Cannot activate unknown/undownloaded model %s", version)
                return False
            task_type = target.get("task_type", TASK_WHISPER_VOICE)
            # Only one active model *per task*; leave other tasks untouched.
            for m in data["models"]:
                if m.get("task_type", TASK_WHISPER_VOICE) == task_type:
                    m["is_active"] = (m.get("version") == version)
            data["active"][task_type] = version
            self._save(data)
        self._mirror_active_to_settings(task_type, version)
        try:
            from src.features import audit_log
            audit_log.log_model_activated(version, target.get("base_model", "base"))
        except Exception as exc:
            logger.debug("Could not write model-activation audit entry: %s", exc)
        logger.info("Activated %s model %s", task_type, version)
        return True

    def rollback_to_base(self, task_type: str = TASK_WHISPER_VOICE) -> None:
        """Deactivate the active model for *task_type*; revert to base weights."""
        with self._lock:
            data = self._load()
            for m in data["models"]:
                if m.get("task_type", TASK_WHISPER_VOICE) == task_type:
                    m["is_active"] = False
            data["active"].pop(task_type, None)
            self._save(data)
        self._mirror_active_to_settings(task_type, None)
        logger.info("Rolled back %s to base model", task_type)

    @staticmethod
    def _mirror_active_to_settings(task_type: str, version: Optional[str]) -> None:
        """Mirror the voice model into settings for cheap reads elsewhere.

        Only the voice model is mirrored (``active_model_version``) for backward
        compatibility; other tasks read the registry directly.
        """
        if task_type != TASK_WHISPER_VOICE:
            return
        try:
            from src.core.settings import Settings
            Settings().set("active_model_version", version)
        except Exception as exc:
            logger.debug("Could not mirror active model into settings: %s", exc)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_active_model_path(
        self, task_type: str = TASK_WHISPER_VOICE
    ) -> Optional[Path]:
        """Return the artifact directory of *task_type*'s active model, or None.

        Returns None (and self-heals the registry) if the active model's
        directory has gone missing on disk.
        """
        data = self._load()
        version = data.get("active", {}).get(task_type)
        if not version:
            return None
        entry = next((m for m in data["models"] if m.get("version") == version), None)
        if not entry or not entry.get("local_path"):
            return None
        path = Path(entry["local_path"])
        if not path.exists():
            logger.warning("Active %s model %s missing on disk; rolling back",
                           task_type, version)
            self.rollback_to_base(task_type)
            return None
        return path

    def list_available_models(
        self, task_type: Optional[str] = None
    ) -> List[ModelVersion]:
        """Registered versions, newest first; optionally filtered by task."""
        data = self._load()
        models = [ModelVersion.from_dict(m) for m in data.get("models", [])]
        if task_type is not None:
            models = [m for m in models if m.task_type == task_type]
        return sorted(models, key=lambda m: m.created_at, reverse=True)

    def get_active_version(
        self, task_type: str = TASK_WHISPER_VOICE
    ) -> Optional[str]:
        return self._load().get("active", {}).get(task_type)
