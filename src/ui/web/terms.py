"""Marked words, their suggestions, and learning from the radiologist's fixes.

Thin on purpose: what counts as suspect, how suggestions rank, and what a fix
teaches all live in ``src.medical.term_lookup`` and ``features/``, so the
browser and the desktop cannot disagree about a word.
"""

from __future__ import annotations

import logging
from dataclasses import asdict

import anyio
from fastapi import APIRouter, HTTPException, Response

from src.core import event_log
from src.devtools import speech_test
from src.features import audit_log
from src.features.adaptive_learning import get_adaptive_learning, learn_from_session
from src.features.edit_tracking import record_session_edits
from src.medical import term_lookup
from src.ui.web import state
from src.ui.web.schemas import ConsentRequest, SessionEditsRequest, TermChoiceRequest, TextRequest

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/api/terms/lookup")
async def term_lookup_endpoint(q: str = ""):
    """The two suggestion lists for a highlighted word or phrase.

    Runs off the event loop because the very first call builds the spelling
    index if start-up warming did not get there first.
    """
    limit = term_lookup.max_query_chars()
    if len(q) > limit:
        raise HTTPException(status_code=422, detail=f"Select at most {limit} characters")
    result = await anyio.to_thread.run_sync(term_lookup.lookup, q)
    return asdict(result)


@router.post("/api/terms/suspect")
async def term_suspect_endpoint(payload: TextRequest):
    """Which words in the report are worth a second look, with their offsets.

    A POST because the body is the whole report, which does not belong in a
    query string.
    """
    keep = frozenset(get_adaptive_learning().custom_terms())
    spans = await anyio.to_thread.run_sync(term_lookup.suspect_terms, payload.text, keep)
    return {"spans": [asdict(span) for span in spans]}


@router.post("/api/terms/used")
async def term_used_endpoint():
    """Record that a suggestion was taken, so the marks hint can retire.

    Counted in settings rather than the browser because the desktop window
    counts the same takes into the same key: learning the feature in one
    front-end should not leave the other still explaining it.
    """
    settings = state.settings()
    ceiling = int(settings.get("term_lookup_hint_uses", 3) or 3)
    used = int(settings.get("term_lookup_uses", 0) or 0)
    if used < ceiling:
        used += 1
        settings.set("term_lookup_uses", used)
    return {"term_lookup_uses": used, "term_lookup_hint_uses": ceiling}


@router.post("/api/learn/consent")
async def learn_consent_endpoint(payload: ConsentRequest):
    """Record the answer to "learn from my corrections?", once per install."""
    settings = state.settings()
    settings.set("learning_enabled", payload.enabled)
    settings.set("learning_consent_shown", True)
    audit_log.log_learning_consent(payload.enabled)
    return {"enabled": payload.enabled}


@router.post("/api/learn/session")
async def learn_session_endpoint(payload: SessionEditsRequest):
    """Learn from how the radiologist corrected a finished dictation.

    Called when a report is committed (export, copy, new report, next
    recording). The diff and the learning rules live in features/; this only
    checks consent.
    """
    if not state.settings().get("learning_enabled", True):
        return {"learned": 0}

    def work() -> int:
        record_session_edits(payload.dictated, payload.final)
        return learn_from_session(payload.dictated, payload.final)

    learned = await anyio.to_thread.run_sync(work)
    if learned:
        event_log.emit("learn", "learned from your edits", corrections=learned)
    return {"learned": learned}


@router.post("/api/learn/choice")
async def learn_choice_endpoint(payload: TermChoiceRequest):
    """Accept a suggestion for a marked word, or keep the word as spoken.

    Accept teaches the correction, so the next dictation fixes it unasked.
    Keep adds the word to the radiologist's own vocabulary, so it is never
    marked again.
    """
    if not state.settings().get("learning_enabled", True):
        return {"learned": False}
    learning = get_adaptive_learning()
    if payload.replacement:
        learned = learning.learn_correction(payload.word, payload.replacement)
    else:
        learning.learn_term(payload.word)
        learned = True
    if learned:
        event_log.emit("learn", "accepted a fix" if payload.replacement else "kept a word",
                       word=payload.word, replacement=payload.replacement)
    return {"learned": learned}


@router.post("/api/dev/speak")
async def dev_speak_endpoint(payload: TextRequest):
    """Typed text spoken as a WAV, for testing dictation without a voice.

    Developer tool (src/devtools/speech_test.py); the page plays the WAV into
    the normal recording path in place of the microphone.
    """
    try:
        wav = await anyio.to_thread.run_sync(speech_test.speak_to_wav, payload.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Test voice failed: %s", exc)
        raise HTTPException(status_code=500, detail="Speech synthesis failed") from exc
    return Response(content=wav, media_type="audio/wav")
