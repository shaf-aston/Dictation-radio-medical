"""
Immutable append-only audit log for radiology report actions.

Every action on a report (create, save, export, clear, critical finding
acknowledgement, learning reset) is written as a JSON-line entry.
The log file is opened in append mode and is never truncated by this module.

Log location: data/audit.log
Retention: files are never deleted automatically — manual archival required
           after the institutional retention period (recommended: 8 years).
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_LOG_FILENAME = "audit.log"
_log_path: Optional[Path] = None


def _get_log_path() -> Path:
    global _log_path
    if _log_path is None:
        from src.features.file_manager import _data_dir
        _log_path = _data_dir() / _LOG_FILENAME
    return _log_path


def _write(action: str, detail: Dict[str, Any]) -> None:
    """Append one JSON-line entry to the audit log."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "action": action,
        **detail,
    }
    try:
        with open(_get_log_path(), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception as exc:
        logger.warning("Audit log write failed: %s", exc)


# ---------------------------------------------------------------------------
# Public API — one function per loggable action
# ---------------------------------------------------------------------------

def log_report_saved(path: str, patient_id: str, word_count: int) -> None:
    _write("report_saved", {
        "path": path,
        "patient_id": patient_id,
        "word_count": word_count,
    })


def log_report_exported(path: str, patient_id: str, fmt: str) -> None:
    _write("report_exported", {
        "path": path,
        "patient_id": patient_id,
        "format": fmt,
    })


def log_report_cleared(patient_id: str) -> None:
    _write("report_cleared", {"patient_id": patient_id})


def log_autosave(path: str, patient_id: str) -> None:
    _write("autosave", {"path": path, "patient_id": patient_id})


def log_template_loaded(template_name: str) -> None:
    _write("template_loaded", {"template": template_name})


def log_critical_finding_acknowledged(term: str, patient_id: str, level: int) -> None:
    _write("critical_finding_acknowledged", {
        "term": term,
        "patient_id": patient_id,
        "level": level,
    })


def log_critical_finding_overridden(terms: str, patient_id: str) -> None:
    """Logged when the radiologist proceeds despite unacknowledged critical findings."""
    _write("critical_finding_overridden", {
        "terms": terms,
        "patient_id": patient_id,
    })


def log_learning_reset() -> None:
    _write("learning_reset", {})


def log_learning_consent(enabled: bool) -> None:
    _write("learning_consent", {"enabled": enabled})


def log_cloud_consent(enabled: bool) -> None:
    """Logged when the user grants/revokes consent to upload training data."""
    _write("cloud_training_consent", {"enabled": enabled})


def log_training_upload(batch_id: str, record_count: int) -> None:
    """Logged when a de-identified training batch is uploaded to Lightning AI."""
    _write("training_upload", {
        "batch_id": batch_id,
        "record_count": record_count,
    })


def log_training_record_dropped(reason: str) -> None:
    """Logged when a correction record is dropped (e.g. failed PHI validation)."""
    _write("training_record_dropped", {"reason": reason})


def log_model_activated(version: str, base_model: str) -> None:
    """Logged when a fine-tuned model version becomes the active transcriber."""
    _write("model_activated", {"version": version, "base_model": base_model})
