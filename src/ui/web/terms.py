"""Marked words, their suggestions, and learning from the radiologist's fixes.

Thin on purpose: what counts as suspect, how suggestions rank, and what a fix
teaches all live in ``src.medical.term_lookup`` and ``features/``, so the
browser and the desktop cannot disagree about a word.
"""

from __future__ import annotations

import logging
from dataclasses import asdict

import anyio
from fastapi import APIRouter, HTTPException

from src.core import event_log
from src.features.adaptive_learning import commit_session, get_adaptive_learning, record_consent
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
    return {"hint_active": term_lookup.note_lookup_used(state.settings())}


@router.post("/api/learn/consent")
async def learn_consent_endpoint(payload: ConsentRequest):
    """Record the answer to "learn from my corrections?", once per install."""
    record_consent(state.settings(), payload.enabled)
    return {"enabled": payload.enabled}


@router.post("/api/learn/session")
async def learn_session_endpoint(payload: SessionEditsRequest):
    """Learn from how the radiologist corrected a finished dictation.

    Called when a report is committed (export, copy, new report, next
    recording). Consent, the diff and the learning rules all live in
    features/adaptive_learning.commit_session.
    """
    learned = await anyio.to_thread.run_sync(commit_session, state.settings(), payload.dictated, payload.final)
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
    if not state.settings().get("learning_enabled"):
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
