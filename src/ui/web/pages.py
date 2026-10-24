"""The page, its assets, and the first-load state the page is built from."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import HTMLResponse

from src.core.paths import templates_dir
from src.core.patient_schema import empty_patient_info
from src.core.settings import get_default
from src.dictation.asr.factory import describe_engine
from src.dictation import mic
from src.dictation.asr.models import SUPPORTED_MODELS
from src.dictation.postprocess.pipeline import CLEANUP_LEVEL_LABELS
from src.features import template_fields
from src.dictation.postprocess.accent import ACCENT_LABELS
from src.features.clinical_disclaimer import DISCLAIMER_TEXT, DISCLAIMER_TITLE, needs_showing
from src.features.report_manager import DOCX_AVAILABLE
from src.medical import term_lookup
from src.features.report_release import COMMUNICATION_METHODS
from src.ui.theme import css_variables
from src.ui.web import state

router = APIRouter()

THEME_STORAGE_KEY = "radio-dictate-theme"

#: The front-end lives in real .html/.css/.js files under ``frontends/`` rather
#: than in a Python string, so it can be edited, linted and diffed like code.
#: Only these are ever served: the name is never taken from a request.
FRONTEND_DIR = Path(__file__).parent.parent / "frontends"
FRONTEND_FILES = {
    "app.html": "text/html; charset=utf-8",
    "app.css": "text/css; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    # Loaded into the AudioWorklet only when the browser cannot give a 16 kHz
    # context on the microphone (Firefox).
    "resampler.js": "text/javascript; charset=utf-8",
    "favicon.svg": "image/svg+xml",
    # The /developer diagnostics page. It reuses app.css for the shared
    # token-based primitives and adds only what a table needs; it deliberately
    # does NOT load app.js, which drives the dictation DOM and would run the
    # whole recording wiring against elements this page does not have.
    "developer.html": "text/html; charset=utf-8",
    "developer.css": "text/css; charset=utf-8",
    "developer.js": "text/javascript; charset=utf-8",
}

#: Pages, not assets: these are rendered through render_html (they carry
#: __THEME_VARS__ placeholders) and would serve unfilled slots over /static/.
_PAGE_FILES = {"app.html", "developer.html"}

#: Nothing the front-end serves may be cached by the browser. See
#: :func:`frontend_file` for why a stale copy is worse than a re-read.
NO_STORE = {"Cache-Control": "no-store"}

#: name -> (mtime_ns, text)
_frontend_cache: dict[str, tuple[int, str]] = {}


def frontend_file(name: str) -> str:
    """Read one bundled front-end file.

    Cached in memory, but keyed on the file's modification time: so editing
    app.js shows up on the next reload instead of needing a server restart.
    The old cache made a stale front-end indistinguishable from a working one:
    the browser went on using the previous release's code against the current
    server, which is how a page kept using the old record-then-upload path
    (and felt many seconds slower) long after live dictation had landed.
    """
    if name not in FRONTEND_FILES:
        raise HTTPException(status_code=404, detail="Not found")
    path = FRONTEND_DIR / name
    stamp = path.stat().st_mtime_ns
    cached = _frontend_cache.get(name)
    if cached is None or cached[0] != stamp:
        cached = (stamp, path.read_text(encoding="utf-8"))
        _frontend_cache[name] = cached
    return cached[1]


def normalize_theme(theme: object) -> str:
    return str(theme) if str(theme) in {"dark", "light"} else "dark"


def current_theme() -> str:
    return normalize_theme(state.settings().get("theme"))


def template_names() -> list[str]:
    """Return bundled template filenames sorted alphabetically."""
    tpl_path = templates_dir()
    if not tpl_path.is_dir():
        return []
    return sorted(
        item.name
        for item in tpl_path.iterdir()
        if item.is_file() and item.suffix.lower() == ".txt"
    )


def bootstrap_payload() -> dict:
    """Collect the page bootstrap state so the client can render controls."""
    settings = state.settings()
    templates = template_names()
    selected_template = settings.get("last_template")
    if selected_template not in templates:
        selected_template = templates[0] if templates else ""
    return {
        # The browser must capture at exactly the rate the decoder expects, so
        # the page is told the number rather than hardcoding its own copy of it.
        "live_sample_rate": state.LIVE_SAMPLE_RATE,
        # Which recogniser actually answers: the model picker only names
        # Whisper, which on the default engine is the fallback.
        "engine_label": describe_engine(settings),
        # The communication dialog's choices and the name it last signed with.
        "communication": {
            "methods": list(COMMUNICATION_METHODS),
            "radiologist": str(settings.get("radiologist_name") or ""),
        },
        # What a template field is, as a pattern valid in JS too: the page
        # finds fields with the server's own definition, not a copy of it.
        "field_pattern": template_fields.FIELD_PATTERN,
        "not_a_field_pattern": template_fields.NOT_A_FIELD_PATTERN,
        # The same key settings the desktop binds (a dictation mic's buttons
        # send these in keyboard mode).
        "keys": {
            name: str(settings.get(f"key_{name}") or get_default(f"key_{name}"))
            for name in ("record_toggle", "next_field", "previous_field")
        },
        "templates": templates,
        "selected_template": selected_template,
        "preferences": state.current_preferences(),
        "patient_defaults": empty_patient_info(),
        "supported_models": SUPPORTED_MODELS,
        "accent_options": [
            {"key": key, "label": label} for key, label in ACCENT_LABELS.items()
        ],
        "cleanup_options": [
            {"key": key, "label": label} for key, label in CLEANUP_LEVEL_LABELS.items()
        ],
        "macros": state.macros_payload(),
        "docx_available": DOCX_AVAILABLE,
        # How many times a marked word's suggestion has actually been taken.
        # The marks hint stops explaining itself past the ceiling; the count
        # lives in settings rather than the browser so using the feature on the
        # desktop also retires the hint here.
        "lookup_hint_active": term_lookup.lookup_hint_active(settings.get),
        "mic": mic.as_payload(),
        # The developer console's scroll-back is the server's ring, not a copy.
        "event_log_max": int(settings.get("event_log_max")),
        # The first-launch clinical disclaimer, shipped with the rest of the
        # first-load state rather than fetched separately: the page must be
        # able to show it before the radiologist can type anything.
        "disclaimer": {
            "title": DISCLAIMER_TITLE,
            "text": DISCLAIMER_TEXT,
            "needed": needs_showing(settings),
        },
        # Learning from the radiologist's fixes: the same two settings the
        # desktop's consent dialog writes, so answering in one front-end
        # answers for both.
        "learning": {
            "enabled": bool(settings.get("learning_enabled")),
            "consent_needed": not settings.get("learning_consent_shown"),
        },
    }


#: JSON inside an inline <script>: the HTML parser reads it before the script
#: does, so "</script>", "<!--" and "&" in a value (a radiologist's name is
#: enough) can end the script or change how it is parsed. Escaped as \uXXXX,
#: which JSON reads back as the same character.
_SCRIPT_UNSAFE = str.maketrans({"<": "\\u003c", ">": "\\u003e", "&": "\\u0026",
                                "\u2028": "\\u2028", "\u2029": "\\u2029"})


def _json_for_script(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).translate(_SCRIPT_UNSAFE)


def render_html(theme: str) -> str:
    """The page shell, with the saved theme and the bootstrap state injected.

    The markup, styles and script are real files under ``frontends/``: only
    two things vary per request, so only two things are substituted.
    """
    bootstrap = _json_for_script(bootstrap_payload())
    script = (
        f'window.__THEME_STORAGE_KEY__={json.dumps(THEME_STORAGE_KEY)};'
        f'window.__BOOTSTRAP__={bootstrap};'
    )
    return (
        frontend_file("app.html")
        .replace("__THEME__", normalize_theme(theme))
        .replace("__THEME_VARS__", css_variables())
        .replace("__BOOTSTRAP_SCRIPT__", script)
    )


@router.get("/static/{name}")
async def static_file(name: str):
    """Serve the bundled front-end. Only the known asset names resolve."""
    media_type = FRONTEND_FILES.get(name)
    if media_type is None or name in _PAGE_FILES:
        raise HTTPException(status_code=404, detail="Not found")
    return Response(
        content=frontend_file(name),
        media_type=media_type,
        # The server is on this machine, so re-reading a few KB costs nothing,
        # and a browser holding yesterday's app.js against today's server is a
        # real failure that looks like a slow app rather than a stale one.
        headers=NO_STORE,
    )


@router.get("/")
async def root():
    return HTMLResponse(render_html(current_theme()), headers=NO_STORE)


@router.get("/developer")
async def developer_page():
    """Local diagnostics: every dictation run, and how each one behaved.

    Unlisted on purpose: no link from the reporting UI. It is for whoever is
    asking why dictation felt slow, not part of writing a report.
    """
    return HTMLResponse(
        frontend_file("developer.html").replace("__THEME_VARS__", css_variables()),
        headers=NO_STORE,
    )


@router.get("/favicon.ico")
async def favicon():
    favicon_path = FRONTEND_DIR / "favicon.svg"
    if favicon_path.exists():
        return Response(content=favicon_path.read_bytes(), media_type="image/svg+xml")
    return Response(content=b"", status_code=204)
