"""Registry of downloaded fine-tuned models — the source of truth for which
model the local Transcriber loads.

State lives in ``data/models/registry.json`` so it is human-inspectable and
survives without the cloud being reachable. Exactly one version may be active;
``get_active_model_path`` returns its CTranslate2 directory, or None to mean
"use the stock base model". Activation is reversible via ``rollback_to_base``.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from src.training.schemas import ModelVersion

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Read/write access to the fine-tuned model index."""

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
        if not self._path.is_file():
            return {"active_version": None, "models": []}
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Could not read model registry: %s", exc)
            return {"active_version": None, "models": []}

    def _save(self, data: dict) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            logger.warning("Could not write model registry: %s", exc)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_downloaded_model(
        self,
        version: str,
        local_path: Path,
        base_model: str,
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
        logger.info("Registered fine-tuned model %s at %s", version, local_path)
        return mv

    # ------------------------------------------------------------------
    # Activation
    # ------------------------------------------------------------------

    def activate_model(self, version: str) -> bool:
        """Make *version* the active model. Returns False if unknown/undownloaded."""
        with self._lock:
            data = self._load()
            target = next((m for m in data["models"] if m.get("version") == version), None)
            if target is None or not target.get("local_path"):
                logger.warning("Cannot activate unknown/undownloaded model %s", version)
                return False
            for m in data["models"]:
                m["is_active"] = (m.get("version") == version)
            data["active_version"] = version
            self._save(data)
        # Mirror into settings so other components can read it cheaply.
        try:
            from src.core.settings import Settings
            Settings().set("active_model_version", version)
        except Exception:
            pass
        try:
            from src.features import audit_log
            audit_log.log_model_activated(version, target.get("base_model", "base"))
        except Exception:
            pass
        logger.info("Activated fine-tuned model %s", version)
        return True

    def rollback_to_base(self) -> None:
        """Deactivate any fine-tuned model; Transcriber reverts to base."""
        with self._lock:
            data = self._load()
            for m in data["models"]:
                m["is_active"] = False
            data["active_version"] = None
            self._save(data)
        try:
            from src.core.settings import Settings
            Settings().set("active_model_version", None)
        except Exception:
            pass
        logger.info("Rolled back to base model")

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_active_model_path(self) -> Optional[Path]:
        """Return the CT2 directory of the active model, or None for base.

        Returns None (and self-heals the registry) if the active model's
        directory has gone missing on disk.
        """
        data = self._load()
        version = data.get("active_version")
        if not version:
            return None
        entry = next((m for m in data["models"] if m.get("version") == version), None)
        if not entry or not entry.get("local_path"):
            return None
        path = Path(entry["local_path"])
        if not path.exists():
            logger.warning("Active model %s missing on disk; rolling back", version)
            self.rollback_to_base()
            return None
        return path

    def list_available_models(self) -> List[ModelVersion]:
        """All registered versions, newest first."""
        data = self._load()
        models = [ModelVersion.from_dict(m) for m in data.get("models", [])]
        return sorted(models, key=lambda m: m.created_at, reverse=True)

    def get_active_version(self) -> Optional[str]:
        return self._load().get("active_version")
