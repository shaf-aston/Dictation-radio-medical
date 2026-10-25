"""The live dictation socket: browser audio in, transcript updates out.

The loop itself (chunking, decoding, the preview, the polish) is
``dictation/stream/live_session.py``, shared with the desktop. This module only
moves bytes and messages between it and the browser, and records the run.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from typing import Optional

import anyio
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.core import event_log, perf
from src.cloud.framework.registry import active_voice_model_path
from src.dictation.stream.live_session import LiveSession
from src.features import run_log, template_fields
from src.ui.web import state

logger = logging.getLogger(__name__)
router = APIRouter()


def build_live_session(settings) -> LiveSession:
    """A session from the saved settings (LiveSession.from_settings reads them).

    The accurate engine uses an active fine-tuned voice model, as the desktop
    does; live text always comes from the stock fast model.
    """
    live_engine = state.get_engine(settings.get("live_model_size"))
    final_engine = state.get_engine(settings.get("model_size"), active_voice_model_path())
    return LiveSession.from_settings(
        settings.get, live_engine, final_engine, sr=state.LIVE_SAMPLE_RATE,
    )


def record_web_run(
    settings, prefs: dict, text: str, session: LiveSession, *,
    elapsed: float, finalise_sec: float = 0.0,
) -> None:
    """Write one run record for a browser dictation, as the desktop does.

    *finalise_sec* is how long after the last word the final text arrived:
    for streaming, that number is the whole point. The word count, stage
    timings, decode ratio and stored text are filled in by run_log.finish,
    the same as for the desktop, so the two rows on /developer mean the same.
    """
    audio_sec = session.audio_sec
    record = run_log.start(
        "web", model=str(prefs.get("model_size", "")), engine=str(settings.get("asr_engine")),
    )
    record.audio_sec = audio_sec
    record.duration_sec = round(elapsed, 3)
    record.finalise_sec = round(finalise_sec, 3)
    record.chunk_count = session.chunks_decoded
    session.record_decode_ratio()
    run_log.finish(record, text, settings)
    # The one line that summarises the whole recording, so the console ends
    # with the verdict rather than leaving it to be added up by eye.
    event_log.emit(
        "run", "recording finished",
        audio_sec=round(audio_sec, 1), wall_sec=round(elapsed, 1),
        polish_sec=round(finalise_sec, 1), words=record.word_count,
        chunks=record.chunk_count,
        words_per_min=round(record.word_count / (audio_sec / 60), 1) if audio_sec > 1 else None,
        wall_per_audio_sec=round(elapsed / audio_sec, 3) if audio_sec > 0 else None,
    )


@router.websocket("/ws/dictate")
async def dictate_socket(ws: WebSocket) -> None:
    await ws.accept()
    generation = state.next_dictation()
    # A new dictation's stage timings start from zero, as on the desktop: the
    # run record snapshots them at the end (run_log.finish).
    perf.reset()
    settings = state.settings()
    prefs = state.current_preferences()
    session = build_live_session(settings)
    cycle_sec = float(settings.get("live_cycle_sec"))
    max_sec = int(settings.get("max_upload_mb")) * 1024 * 1024 / (2 * state.LIVE_SAMPLE_RATE)

    def superseded() -> bool:
        return generation != state.dictation_generation

    started = time.time()
    last_cycle = 0.0
    # Everything below feeds the developer panel's console. The one number a
    # radiologist actually feels is "how long from speaking to seeing it", so
    # it is measured explicitly rather than inferred from a rolling mean.
    first_words_at: Optional[float] = None
    cycles = 0
    event_log.emit("live", "dictation socket open", cycle_sec=cycle_sec)
    # A decode (the preview or a closing chunk) costs whole seconds on this
    # machine (see should_skip_preview's docstring): awaiting it here before
    # looping back to receive() would stall reading the socket for that long,
    # and the microphone does not pause while it waits. Running each cycle as
    # a background task instead means incoming audio is always drained
    # immediately; only the transcript's freshness lags behind, not the
    # capture of the audio itself. `cycle_task` guards against two decodes
    # running at once, which session.cycle() is not written to survive.
    cycle_task: Optional[asyncio.Task] = None
    # The template field this dictation replaces, if the radiologist selected
    # one. Saying one of a pick-list's options answers it in the template's own
    # spelling: the same rule the desktop applies (template_fields.resolve_choice).
    field_text: Optional[str] = None

    def answered(text: str) -> str:
        if field_text:
            return template_fields.resolve_choice(field_text, text) or text
        return text

    async def send(payload: dict) -> None:
        try:
            await ws.send_json(payload)
        except Exception:
            pass  # the page navigated away mid-send; the finally block cleans up

    async def run_cycle() -> None:
        nonlocal first_words_at, cycles
        with event_log.timed("live", "decode cycle", stage="web.cycle") as note:
            update = await anyio.to_thread.run_sync(session.cycle)
            cycles += 1
            note["cycle"] = cycles
            if update is None:
                note["result"] = "nothing new"
                return
            words = len(update.committed.split())
            note.update(
                audio_sec=round(update.audio_sec, 1),
                # How far the text on screen is behind the microphone. This is
                # the lag the radiologist sees; the decode cost above is only
                # the reason for it.
                behind_sec=round(max(0.0, time.time() - started - update.audio_sec), 1),
                committed_words=words,
                preview_words=len(update.preview.split()),
                unsure_words=len(update.uncertain),
                state=update.state,
            )
            if first_words_at is None and (words or update.preview):
                first_words_at = time.time()
                event_log.emit(
                    "live", "first words on screen",
                    ms=(first_words_at - started) * 1000,
                )
            await send({
                "type": "partial",
                "committed": answered(update.committed),
                "preview": update.preview,
                "state": update.state,
                "audioSec": round(update.audio_sec, 1),
                # Where the frozen text ends, in audio seconds. The ribbon
                # draws one block per chunk from this: without it the browser
                # had to guess a chunk had closed from the word count growing,
                # which drew every update as another block and turned the
                # session's shape into one flat slab.
                "savedSec": round(session.committed_sec, 1),
                "uncertain": list(update.uncertain),
            })

    try:
        while True:
            message = await ws.receive()

            if message.get("type") == "websocket.disconnect":
                return

            if (raw := message.get("bytes")) is not None:
                # Reject an over-long recording rather than letting one
                # connection grow unbounded.
                if session.audio_sec > max_sec:
                    await send({"type": "error", "message": "Recording too long. Please stop and start a new one."})
                    return
                session.feed(np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0)

                if time.time() - last_cycle >= cycle_sec and (cycle_task is None or cycle_task.done()):
                    last_cycle = time.time()
                    cycle_task = asyncio.create_task(run_cycle())
                continue

            if (text := message.get("text")) is None:
                continue

            try:
                request = json.loads(text)
                command = request.get("command")
            except (ValueError, AttributeError):
                continue
            if command == "field":
                field_text = str(request.get("text") or "")[:400] or None
                continue
            if command == "stop":
                stop_at = time.time()
                event_log.emit(
                    "live", "stop pressed",
                    audio_sec=round(session.audio_sec, 1), chunks=session.chunks_decoded,
                )
                # A cycle may still be decoding a chunk the ledger hasn't
                # committed yet: finalize() must see that commit, not race it.
                if cycle_task is not None and not cycle_task.done():
                    with event_log.timed("live", "waited for the in-flight decode"):
                        await cycle_task
                # Close the still-open tail with the fast model first. Without
                # this the hand-back is everything except the last chunk, and
                # the last chunk is where the impression lives: a report that
                # says "ready to edit" while a third of it is still missing is
                # worse than one that took a second longer to arrive.
                with event_log.timed("live", "closing the last section", stage="web.stop_tail"):
                    await anyio.to_thread.run_sync(session.close_open_tail_fast)
                # Now hand it back. The radiologist has been reading this text
                # as they spoke it, so it is theirs to edit: the accurate
                # re-decode below is an upgrade, not a gate, and blocking on it
                # would put the old wait straight back.
                await send({
                    "type": "stopped",
                    "text": answered(session.committed_text()),
                    "uncertain": list(session.uncertain_words),
                })
                audio_ended = time.time()
                event_log.emit(
                    "live", "report handed back",
                    ms=(audio_ended - stop_at) * 1000,
                    words=len(session.committed_text().split()),
                )
                with event_log.timed("live", "accuracy pass", stage="web.finalize") as note:
                    text_out = await anyio.to_thread.run_sync(
                        lambda: session.finalize(cancelled=superseded)
                    )
                    note["words"] = len(text_out.split())
                    note["unsure_words"] = len(session.uncertain_words)
                if superseded():
                    # The report was already handed back; a newer recording owns the page now.
                    event_log.emit("live", "polish dropped: a new recording started")
                    return
                await send({
                    "type": "final",
                    "text": answered(text_out),
                    "uncertain": list(session.uncertain_words),
                })
                record_web_run(
                    settings, prefs, text_out, session,
                    elapsed=time.time() - started,
                    finalise_sec=time.time() - audio_ended,
                )
                return
            if command == "cancel":
                event_log.emit("live", "recording discarded", audio_sec=round(session.audio_sec, 1))
                return
    except WebSocketDisconnect:
        return
    except Exception as exc:
        logger.error("Live dictation failed: %s", exc, exc_info=True)
        event_log.emit("live", f"dictation failed: {exc}", level="error")
        await send({"type": "error", "message": "Dictation stopped. The text so far is kept: record again to continue."})
    finally:
        if cycle_task is not None and not cycle_task.done():
            cycle_task.cancel()
        # The engine's live socket, if one is open: closing joins its threads,
        # so it is done off the event loop.
        with contextlib.suppress(Exception):
            await anyio.to_thread.run_sync(session.close)
        with contextlib.suppress(Exception):
            await ws.close()
