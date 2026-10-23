"""What the developer console and /developer read.

Local only: these read in-process buffers and a file under ``data/``, served
on loopback. Nothing is stored by them and nothing leaves the device.
"""

from __future__ import annotations

from fastapi import APIRouter

from src.core import event_log, perf
from src.features import run_log

router = APIRouter()


@router.get("/api/runs")
async def runs_endpoint(limit: int = 100):
    """Every recorded dictation run, newest first (see features/run_log)."""
    limit = max(1, min(int(limit), 500))
    return {"runs": run_log.recent(limit=limit)}


@router.get("/api/debug/perf")
async def perf_endpoint():
    """Rolling stage timings for this process: where dictation time goes."""
    return {"stages": perf.snapshot(), "gauges": perf.gauges()}


@router.get("/api/debug/events")
async def events_endpoint(after: int = 0, limit: int = 500):
    """The live diary: what happened since event *after*, oldest first.

    The developer panel passes back the last sequence number it printed, so a
    poll only ever carries what is new.
    """
    return {
        "events": event_log.events(after=max(0, int(after)), limit=max(1, min(int(limit), 2000))),
        "latest": event_log.latest_seq(),
        "running": event_log.running(),
    }
