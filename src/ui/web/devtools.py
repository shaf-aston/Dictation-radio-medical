"""The developer Test voice's endpoint (src/devtools/speech_test.py).

Registered by web_app only when ``src.devtools`` is present, so removing the
tool is deleting that folder, this file and ``frontends/speechtest.js``.
"""

from __future__ import annotations

import logging

import anyio
from fastapi import APIRouter, HTTPException, Response

from src.devtools import speech_test
from src.ui.web.schemas import TextRequest

logger = logging.getLogger(__name__)
router = APIRouter()


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
