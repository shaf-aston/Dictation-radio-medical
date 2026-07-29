"""One record per dictation run: how long it took, and whether it kept up.

``src.core.perf`` already measures where dictation time goes, but it is
in-process and resets with every recording — so "did that twenty-minute
dictation behave differently from this two-minute one?" could only ever be
asserted, never shown. This module is what turns those timings into history,
and ``/developer`` is where they are read.

Shape follows ``features/audit_log.py``: append-only JSON Lines through
``core/json_store``. Two things differ, and both are because this is
diagnostics rather than an institutional record:

* **It is capped and rolls.** The audit log is never auto-deleted; this one
  keeps the most recent ``max_runs`` and drops the rest. It holds a full report
  per run, so an uncapped file would grow without limit on the one machine that
  can least afford it.
* **The report text is optional.** ``run_log_store_text`` (default on, because
  seeing the output per run is the point) writes the report alongside the
  numbers. Turning it off keeps every timing and drops only the text — for a
  site that would rather no report body sat in a diagnostics file.

Local only. Nothing here is uploaded, and nothing calls out; the offline
invariant is untouched.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.core import perf
from src.core.json_store import append_jsonl, read_jsonl
from src.features.file_manager import run_log_path

logger = logging.getLogger(__name__)

#: Kept if settings carry no value. Bounded so the file cannot grow forever.
DEFAULT_MAX_RUNS = 200

#: How much the file may overshoot the cap before it is rewritten. Trimming on
#: every append would rewrite the whole log per run; this amortises it.
_TRIM_SLACK = 50


@dataclass
class RunRecord:
    """One dictation, from pressing record to the final text being applied.

    Built empty at the start of a run and filled in as the facts arrive, so a
    run that crashes half-way still has a beginning to write.
    """

    started: str
    front_end: str                       # "desktop" | "web"
    model: str = ""
    engine: str = ""
    audio_sec: float = 0.0
    #: Wall-clock from pressing record to the final text landing.
    duration_sec: float = 0.0
    #: How long after the audio ended the final text arrived. The number that
    #: says whether Stop feels instant or not.
    finalise_sec: float = 0.0
    word_count: int = 0
    chunk_count: int = 0
    #: Decoded seconds per second of audio. Above 1.0 the machine cannot keep
    #: up with speech. Read from perf's gauge, not recomputed here.
    decode_ratio: Optional[float] = None
    stages: Dict[str, Dict[str, float]] = field(default_factory=dict)
    text: str = ""


def _cap(settings: Any) -> int:
    try:
        value = int(settings.get("run_log_max", DEFAULT_MAX_RUNS))
    except (TypeError, ValueError):
        return DEFAULT_MAX_RUNS
    return max(1, value)


def _wants_text(settings: Any) -> bool:
    return bool(settings.get("run_log_store_text", True))


def start(front_end: str, model: str = "", engine: str = "") -> RunRecord:
    """Open a record. Call when recording starts, beside ``perf.reset()``."""
    return RunRecord(
        started=datetime.now(timezone.utc).isoformat(),
        front_end=front_end,
        model=model,
        engine=engine,
    )


def finish(record: RunRecord, text: str, settings: Any) -> RunRecord:
    """Fill in what is only known at the end, then write the record.

    Takes the perf snapshot *here* rather than letting the caller pass one,
    because the next run's ``perf.reset()`` is what makes these numbers
    unrecoverable — capturing them at close is the whole point.
    """
    record.word_count = len(text.split())
    record.stages = perf.snapshot()
    record.decode_ratio = perf.gauges().get("stream.decode_ratio")
    record.text = text if _wants_text(settings) else ""
    write(record, settings)
    return record


def write(record: RunRecord, settings: Any) -> None:
    """Append *record*, trimming the log when it has drifted past the cap.

    Never raises: a diagnostics log must not be able to break the dictation it
    is describing.
    """
    try:
        path = run_log_path()
        append_jsonl(path, [asdict(record)])
        cap = _cap(settings)
        rows = read_jsonl(path)
        if len(rows) > cap + _TRIM_SLACK:
            _rewrite(rows[-cap:])
    except Exception as exc:
        logger.warning("Could not record the dictation run: %s", exc)


def _rewrite(rows: List[Dict[str, Any]]) -> None:
    """Replace the log with *rows*, atomically.

    Written to a sibling and renamed over the original, so a crash mid-trim
    leaves the old complete log rather than a half-written one.
    """
    path = run_log_path()
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def recent(limit: int = 0) -> List[Dict[str, Any]]:
    """Every recorded run, newest first. ``limit`` 0 means all of them."""
    rows = read_jsonl(run_log_path())
    rows.reverse()
    return rows[:limit] if limit > 0 else rows


def clear() -> None:
    """Delete the log. Diagnostics, so this is always safe."""
    run_log_path().unlink(missing_ok=True)
