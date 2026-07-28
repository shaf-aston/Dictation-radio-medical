"""The one home for the first-launch clinical disclaimer.

Says what the disclaimer is and whether this user has seen it yet; nothing about
how it is shown. Both front-ends call it: the desktop shows a message box, the
web app ships the text in its page bootstrap and takes the acknowledgement over
HTTP. Neither owns the wording, so neither can drift from the other — this
module has no Qt, no HTTP and no dialog vocabulary in it.

The wording is a liability statement, not copy. Editing it is a legal decision,
not a refactor.

Not to be confused with ``src/imaging/schemas.py::DISCLAIMER`` — that one is the
scan assistant's per-result "assistive, not a diagnosis" notice. This one is the
app-wide first-launch statement about speech recognition. Two statements, two
scopes; keep them apart.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.settings import Settings

logger = logging.getLogger(__name__)

#: The heading — a window title on the desktop, the modal's heading on the web.
DISCLAIMER_TITLE = "Clinical Disclaimer"

DISCLAIMER_TEXT = (
    "DISCLAIMER — IMPORTANT\n\n"
    "This tool uses OpenAI Whisper for speech recognition. Whisper is a "
    "general-purpose model and is not FDA-cleared or CE-marked for clinical "
    "medical documentation.\n\n"
    "All transcriptions MUST be reviewed and verified by a qualified "
    "radiologist before clinical use or patient record entry.\n\n"
    "The software author accepts no liability for transcription errors."
)

#: Where the "they have seen it" answer is kept. Shared by both front-ends, so
#: acknowledging in one settles it for the other.
_SETTING = "disclaimer_shown"


def needs_showing(settings: Settings) -> bool:
    """True until this user has acknowledged the disclaimer."""
    return not settings.get(_SETTING, False)


def mark_shown(settings: Settings) -> None:
    """Record that the disclaimer was acknowledged. Calling it again is harmless."""
    if needs_showing(settings):
        settings.set(_SETTING, True)
