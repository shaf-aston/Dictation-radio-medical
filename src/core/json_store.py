"""Shared JSON-file read/write helpers.

``Settings``, ``ModelRegistry``, ``DatasetRegistry``, the imaging threshold
loader, adaptive learning, macros, the audit log, and edit tracking each
hand-rolled the same "read JSON with a fallback default; write JSON, creating
parent directories" skeleton — including its error handling, which is the part
that was easiest to get subtly wrong. This module is that skeleton, written
once: whole-document JSON (:func:`read_json` / :func:`write_json`) and
append-only JSON Lines (:func:`append_jsonl` / :func:`read_jsonl`).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

logger = logging.getLogger(__name__)


def read_json(path: Path, default: Any) -> Any:
    """Return the JSON contents of *path*, or *default* if missing/unreadable."""
    path = Path(path)
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return default


def write_json(path: Path, data: Any) -> None:
    """Write *data* as indented JSON to *path*, creating parent directories.

    Written atomically (temp file + ``os.replace``) so a crash mid-write can
    never leave a truncated file — a corrupt registry/settings file is silently
    read back as the empty default, losing the active fine-tuned-model pointer.
    """
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    except (OSError, TypeError) as exc:
        logger.warning("Could not write %s: %s", path, exc)


def append_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    """Append *records* to *path*, one JSON object per line.

    Used by append-only logs (audit trail, dictation edits) where rewriting the
    whole document per entry would be wasteful. Returns the number of records
    written; logging must never break the operation it is recording, so a
    failure is warned about and reported as ``0``, not raised.
    """
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with open(path, "a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
        return written
    except (OSError, TypeError) as exc:
        logger.warning("Could not append to %s: %s", path, exc)
        return 0


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSON Lines file, oldest first. Missing/unreadable → ``[]``.

    Malformed lines are skipped rather than failing the whole read: a log
    truncated by a crash should still yield the records that did land.
    """
    path = Path(path)
    if not path.is_file():
        return []
    records: List[Dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed line in %s", path)
    except OSError as exc:
        logger.warning("Could not read %s: %s", path, exc)
    return records
