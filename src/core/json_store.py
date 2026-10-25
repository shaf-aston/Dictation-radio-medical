"""Shared JSON-file read/write helpers.

``Settings``, ``ModelRegistry``, ``DatasetRegistry``, the imaging threshold
loader, adaptive learning, macros, the audit log, and edit tracking each
hand-rolled the same "read JSON with a fallback default; write JSON, creating
parent directories" skeleton: including its error handling, which is the part
that was easiest to get subtly wrong. This module is that skeleton, written
once: whole-document JSON (:func:`read_json` / :func:`write_json`, and
:func:`update_json` for a change that must not lose another process's) and
append-only JSON Lines (:func:`append_jsonl` / :func:`read_jsonl`).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from src.core.file_lock import FileLock

logger = logging.getLogger(__name__)


def read_json(path: Path, default: Any) -> Any:
    """Return the JSON contents of *path*, or *default* if missing/unreadable.

    A file that exists but does not parse is moved aside first, to
    ``<name>.corrupt-<time>``. Every caller writes back what it read, so
    returning *default* with the file left in place meant the next save
    replaced someone's whole settings file with one key (a single trailing
    comma was enough). Moved aside, the only copy survives to be repaired.
    """
    path = Path(path)
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:  # unreadable right now (locked, permissions): leave it be
        logger.warning("Could not read %s: %s", path, exc)
        return default
    except json.JSONDecodeError as exc:
        kept = path.with_name(f"{path.name}.corrupt-{time.strftime('%Y%m%d-%H%M%S')}")
        try:
            os.replace(path, kept)
            logger.error("%s is not valid JSON (%s); moved to %s, using defaults", path, exc, kept)
        except OSError as move_exc:
            logger.error("%s is not valid JSON (%s) and could not be moved aside: %s",
                         path, exc, move_exc)
        return default


def write_json(path: Path, data: Any) -> bool:
    """Write *data* as indented JSON to *path*, creating parent directories.

    Written atomically (temp file + ``os.replace``) so a crash mid-write can
    never leave a truncated file: a corrupt registry/settings file is silently
    read back as the empty default, losing the active fine-tuned-model pointer.

    The temp file gets a name unique to this writer rather than ``<target>.tmp``.
    These files have more than one writer: the desktop app and the web app run
    against the same ``dictation_settings.json``, which is why ``Settings`` has a
    ``refresh()`` at all. With a single shared temp name two writers collide:
    the second ``os.replace`` finds the file already consumed, or, worse, moves a
    half-written temp over the target. That is the exact corruption this function
    exists to prevent, so a shared name would defeat its own purpose.

    Returns whether the file was written; a failure is logged, not raised.
    """
    path = Path(path)
    tmp: Optional[str] = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
        tmp = None  # renamed, so there is nothing left to clean up
        return True
    except (OSError, TypeError) as exc:
        logger.warning("Could not write %s: %s", path, exc)
        return False
    finally:
        if tmp is not None:
            # A failed write must not leave a stray temp file next to the real one.
            try:
                os.unlink(tmp)
            except OSError:
                pass


def update_json(path: Path, default: Any, change: Callable[[Any], None]) -> Optional[Any]:
    """Read *path*, let *change* edit the data in place, write it back; return it.

    Returns None when the file could not be written, so the caller still holds
    the change and can try again.

    Read and write happen under one cross-process lock. The desktop app and
    the web app write the same files; without it, each could read, both
    change, and the second write would undo the first.
    """
    path = Path(path)
    with FileLock(path.with_name(f".{path.name}.lock")):
        data = read_json(path, default)
        change(data)
        return data if write_json(path, data) else None


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
