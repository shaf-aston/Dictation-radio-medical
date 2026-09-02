import asyncio
import contextlib
import io
import json

import anyio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from typing import Literal, Optional

from fastapi import FastAPI, File, HTTPException, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.core import event_log, perf
from src.core.patient_schema import PATIENT_KEYS, empty_patient_info
from src.core.settings import Settings, get_default
from src.dictation.postprocess.pipeline import (
    CLEANUP_LEVEL_LABELS,
    CLEANUP_LEVELS,
    postprocess_transcript,
)
from src.dictation.asr import AsrEngine, TranscribeContext, create_engine
from src.dictation.stream.live_session import LiveSession
from src.dictation.stream.rules import build_context_prompt
from src.dictation.stream.segmenter import ChunkPolicy
from src.dictation.transcriber import SUPPORTED_MODELS, resolve_model
from src.features.accent_corrections import ACCENT_LABELS
from src.features.clinical_disclaimer import (
    DISCLAIMER_TEXT,
    DISCLAIMER_TITLE,
    mark_shown,
    needs_showing,
)
from src.features.file_manager import (
    report_filename,
    settings_file,
    startup_cleanup,
    templates_dir,
)
from src.features.report_manager import DOCX_AVAILABLE, export_to_word_bytes, format_plain_text_report
from src.features.report_release import (
    OutstandingFindings,
    record_release,
    unfilled_fields,
)
from src.features import run_log
from src.medical import macros, term_lookup
from src.medical.macros import reload_macros
from src.ui.theme import css_variables

logger = logging.getLogger(__name__)

#: What the browser records at and what Whisper expects. One number,
#: handed to the page in the bootstrap payload so the two cannot drift.
LIVE_SAMPLE_RATE = 16000

transcriber_lock = Lock()
# One engine per model name: the live model and the final model coexist.
asr_engines: dict = {}

THEME_STORAGE_KEY = "radio-dictate-theme"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


class ThemeUpdate(BaseModel):
    theme: Literal["dark", "light"]


class PreferencesUpdate(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_size: str
    language: str = "en"
    vad_filter: bool = True
    accent: str = "neutral"
    cleanup_level: str = "medium"
    macro_region: str = ""


class PatientInfo(BaseModel):
    name: str = ""
    id: str = ""
    dob: str = ""
    study_date: str = ""
    referring: str = ""
    accession: str = ""


# The API contract must not drift from the fields the de-identifier scrubs and
# the report header prints. Adding a field to the shared schema without adding
# it here fails at import, loudly, rather than silently shipping an
# un-scrubbed identifier to the transport layer.
if tuple(PatientInfo.model_fields) != PATIENT_KEYS:
    raise RuntimeError(
        f"PatientInfo fields {tuple(PatientInfo.model_fields)} do not match the "
        f"shared patient schema {PATIENT_KEYS}"
    )


class TextRequest(BaseModel):
    """Just the report text, for the endpoints that only read it."""

    text: str = ""


class ReportRequest(BaseModel):
    text: str = ""
    patient: PatientInfo = Field(default_factory=PatientInfo)
    #: None = the client has not yet been shown the critical-findings warning.
    #: True = radiologist confirmed verbal communication; False = proceeded anyway.
    #: Both outcomes are audited; only None is refused (see _confirm_release).
    acknowledged: Optional[bool] = None
    #: The answer to the unfilled-fields question, which is a plain permission
    #: rather than an audited one: only True releases the report. Not asked yet
    #: and "no, take me back" are the same thing here, so both are False.
    proceed_unfilled: bool = False


def _model_to_dict(model: BaseModel) -> dict:
    """Return model data across Pydantic versions."""
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


def _normalize_theme(theme: object) -> str:
    return str(theme) if str(theme) in {"dark", "light"} else "dark"


#: Settings keys the browser client mirrors. The *values* come from
#: ``settings._DEFAULTS``: re-listing them here is how the web and desktop
#: defaults drifted apart.
_PREFERENCE_KEYS = (
    "model_size", "language", "vad_filter", "accent", "cleanup_level",
)

_settings_instance: Optional[Settings] = None


def _settings() -> Settings:
    """The process-wide settings object.

    Previously a fresh ``Settings()`` per call, which re-read and re-parsed the
    JSON file from disk on every request *and* several times per page render.
    Cached and refreshed only when the file actually changes, so an edit made in
    the desktop app is still picked up. The cache is rebuilt if the settings
    *path* itself changes underneath us.
    """
    global _settings_instance
    path = settings_file()
    if _settings_instance is None or _settings_instance.path != path:
        _settings_instance = Settings()
    else:
        _settings_instance.refresh()
    return _settings_instance


def _current_theme() -> str:
    return _normalize_theme(_settings().get("theme", "dark"))


def _current_preferences() -> dict:
    settings = _settings()
    prefs = {key: settings.get(key, get_default(key)) for key in _PREFERENCE_KEYS}
    macro_region = settings.get("last_macro_region", "")
    if macro_region not in macros.REGION_ORDER and macros.REGION_ORDER:
        macro_region = macros.REGION_ORDER[0]
    prefs["macro_region"] = macro_region
    return prefs


def _template_names() -> list[str]:
    """Return bundled template filenames sorted alphabetically."""
    tpl_path = templates_dir()
    if not tpl_path.is_dir():
        return []
    return sorted(
        item.name
        for item in tpl_path.iterdir()
        if item.is_file() and item.suffix.lower() == ".txt"
    )


#: The front-end lives in real .html/.css/.js files next to this module rather
#: than in a Python string, so it can be edited, linted and diffed like code.
#: Only these three are ever served: the name is never taken from a request.
FRONTEND_DIR = Path(__file__).parent / "frontends"
_FRONTEND_FILES = {
    "app.html": "text/html; charset=utf-8",
    "app.css": "text/css; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "favicon.svg": "image/svg+xml",
    # The /developer diagnostics page. It reuses app.css for the shared
    # token-based primitives and adds only what a table needs; it deliberately
    # does NOT load app.js, which drives the dictation DOM and would run the
    # whole recording wiring against elements this page does not have.
    "developer.html": "text/html; charset=utf-8",
    "developer.css": "text/css; charset=utf-8",
    "developer.js": "text/javascript; charset=utf-8",
}

#: Pages, not assets: these are rendered through _render_html (they carry
#: __THEME_VARS__ placeholders) and would serve unfilled slots over /static/.
_PAGE_FILES = {"app.html", "developer.html"}


def _frontend_file(name: str) -> str:
    """Read one bundled front-end file.

    Cached in memory, but keyed on the file's modification time: so editing
    app.js shows up on the next reload instead of needing a server restart.
    The old cache made a stale front-end indistinguishable from a working one:
    the browser went on using the previous release's code against the current
    server, which is how a page kept using the old record-then-upload path
    (and felt many seconds slower) long after live dictation had landed.
    """
    if name not in _FRONTEND_FILES:
        raise HTTPException(status_code=404, detail="Not found")
    path = FRONTEND_DIR / name
    stamp = path.stat().st_mtime_ns
    cached = _frontend_cache.get(name)
    if cached is None or cached[0] != stamp:
        cached = (stamp, path.read_text(encoding="utf-8"))
        _frontend_cache[name] = cached
    return cached[1]


#: Nothing the front-end serves may be cached by the browser. See
#: :func:`_frontend_file` for why a stale copy is worse than a re-read.
_NO_STORE = {"Cache-Control": "no-store"}

#: name -> (mtime_ns, text)
_frontend_cache: dict[str, tuple[int, str]] = {}


def _macros_payload() -> dict:
    """Return the current macro regions and phrases in JSON-friendly form."""
    regions = [
        {
            "name": region,
            "phrases": [
                {"label": label, "text": text}
                for label, text in macros.MACROS.get(region, [])
            ],
        }
        for region in macros.REGION_ORDER
    ]
    return {
        "regions": regions,
        "selected_region": _current_preferences()["macro_region"],
    }


def _bootstrap_payload() -> dict:
    """Collect the page bootstrap state so the client can render controls."""
    templates = _template_names()
    selected_template = _settings().get("last_template", "")
    if selected_template not in templates:
        selected_template = templates[0] if templates else ""
    return {
        # The browser must capture at exactly the rate the decoder expects, so
        # the page is told the number rather than hardcoding its own copy of it.
        "live_sample_rate": LIVE_SAMPLE_RATE,
        "templates": templates,
        "selected_template": selected_template,
        "preferences": _current_preferences(),
        "patient_defaults": empty_patient_info(),
        "supported_models": SUPPORTED_MODELS,
        "accent_options": [
            {"key": key, "label": label}
            for key, label in ACCENT_LABELS.items()
        ],
        "cleanup_options": [
            {"key": key, "label": label}
            for key, label in CLEANUP_LEVEL_LABELS.items()
        ],
        "macros": _macros_payload(),
        "docx_available": DOCX_AVAILABLE,
        # How many times a marked word's suggestion has actually been taken.
        # The marks hint stops explaining itself past the ceiling; the count
        # lives in settings rather than the browser so using the feature on the
        # desktop also retires the hint here.
        "term_lookup_uses": int(_settings().get("term_lookup_uses", 0) or 0),
        "term_lookup_hint_uses": int(_settings().get("term_lookup_hint_uses", 3) or 3),
        # The first-launch clinical disclaimer, shipped with the rest of the
        # first-load state rather than fetched separately: the page must be
        # able to show it before the radiologist can type anything.
        #
        # There is deliberately no learning-consent equivalent here: adaptive
        # learning is captured only by the desktop window (main_window.py
        # `flush_dictation_edits` / `track_edit`), so the web app collects
        # nothing to consent to. Wiring capture into this front-end means
        # adding the consent gate with it.
        "disclaimer": {
            "title": DISCLAIMER_TITLE,
            "text": DISCLAIMER_TEXT,
            "needed": needs_showing(_settings()),
        },
    }


def _json_for_script(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def _resolve_template_path(template_name: str) -> Path:
    """Resolve a template file safely within the templates directory."""
    base = templates_dir().resolve()
    safe_name = Path(template_name).name
    if not safe_name:
        raise HTTPException(status_code=404, detail="Template not found")

    candidate = (base / safe_name).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Template not found") from exc

    if candidate.suffix.lower() != ".txt" or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Template not found")
    return candidate


def _render_html(theme: str) -> str:
    """The page shell, with the saved theme and the bootstrap state injected.

    The markup, styles and script are real files under ``frontends/``: only
    two things vary per request, so only two things are substituted. The
    template list is no longer injected as markup: it is already in the
    bootstrap payload, and the page builds the options from there.
    """
    bootstrap = _json_for_script(_bootstrap_payload())
    script = (
        f'window.__THEME_STORAGE_KEY__={json.dumps(THEME_STORAGE_KEY)};'
        f'window.__BOOTSTRAP__={bootstrap};'
    )
    return (
        _frontend_file("app.html")
        .replace("__THEME__", _normalize_theme(theme))
        .replace("__THEME_VARS__", css_variables())
        .replace("__BOOTSTRAP_SCRIPT__", script)
    )


def _get_engine(model_size: Optional[str] = None) -> AsrEngine:
    """Return an ASR engine for *model_size* (default: the saved setting).

    Engines are cached per model name, so the live dictation loop's fast model
    and the accurate one that re-decodes after Stop are each built once and
    then reused for the life of the process. Whisper's own weights cache holds
    two (``transcriber._MODEL_CACHE_MAX``), which is what makes holding exactly
    this pair free.
    """
    name = resolve_model(model_size or _settings().get("model_size"))
    with transcriber_lock:
        if name not in asr_engines:
            asr_engines[name] = create_engine(model_size=name)
        return asr_engines[name]


def _download_filename(patient: dict, ext: str) -> str:
    """Attachment filename for a report download (shared builder, sanitised)."""
    return report_filename(
        patient, f".{ext}", prefix="dictation_", fallback_id="unknown"
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Prepare the app state on startup."""
    # Start the diary first, so the warm-up lines below are the first thing the
    # developer panel can show. A panel that only starts recording once you
    # open it can never explain the slow start you opened it to look at.
    event_log.configure(int(_settings().get("event_log_max", get_default("event_log_max"))))
    event_log.attach_to_logging()
    logger.info("Preparing web app state...")
    asr_engines.clear()
    # Same startup housekeeping as the desktop GUI (temp files, old
    # autosaves, legacy cache locations): a web-only user must not miss it.
    startup_cleanup(_settings().get("autosave_retention_days", 30))
    _warm_up_singletons()
    yield
    logger.info("Shutting down")


def _warm_up_singletons() -> None:
    """Pre-build the slow spelling index and Whisper model off the request path.

    Left lazy, both are built on the *first* ``/transcribe`` call and stall it by
    seconds. Warming at startup on a background thread makes the first real
    request fast. Best-effort: warm-up never blocks or fails app startup.
    """
    from src.dictation.warmup import (
        postprocess_warmers,
        transcriber_warmer,
        warm_up_async,
    )

    settings = _settings()
    model_size = resolve_model(settings.get("model_size"))
    live_model_size = resolve_model(settings.get("live_model_size"))
    warmers = postprocess_warmers() + [transcriber_warmer(model_size)]
    # The live model decodes every word shown while the radiologist is still
    # speaking: leaving it lazy means the FIRST dictation after every
    # restart pays its full model-load cost on that exact path, so the first
    # sentence stalls for several seconds before any text appears.
    if live_model_size != model_size:
        warmers.append(transcriber_warmer(live_model_size))
    warm_up_async(warmers)


app = FastAPI(title="Radio Dictate Web", lifespan=lifespan)

@app.get("/static/{name}")
async def static_file(name: str):
    """Serve the bundled front-end. Only the known asset names resolve."""
    media_type = _FRONTEND_FILES.get(name)
    if media_type is None or name in _PAGE_FILES:
        raise HTTPException(status_code=404, detail="Not found")
    return Response(
        content=_frontend_file(name),
        media_type=media_type,
        # The server is on this machine, so re-reading a few KB costs nothing,
        # and a browser holding yesterday's app.js against today's server is a
        # real failure that looks like a slow app rather than a stale one.
        headers=_NO_STORE,
    )




@app.get("/")
async def root():
    return HTMLResponse(_render_html(_current_theme()), headers=_NO_STORE)


@app.get("/developer")
async def developer_page():
    """Local diagnostics: every dictation run, and how each one behaved.

    Unlisted on purpose: no link from the reporting UI. It is for whoever is
    asking why dictation felt slow, not part of writing a report.
    """
    return HTMLResponse(
        _frontend_file("developer.html").replace("__THEME_VARS__", css_variables()),
        headers=_NO_STORE,
    )


@app.get("/favicon.ico")
async def favicon():
    favicon_path = Path(__file__).parent / "frontends" / "favicon.svg"
    if favicon_path.exists():
        return Response(content=favicon_path.read_bytes(), media_type="image/svg+xml")
    return Response(content=b"", status_code=204)


@app.post("/api/disclaimer/ack")
async def acknowledge_disclaimer():
    """Record that the radiologist has read the first-launch disclaimer.

    The HTTP shape of the shared statement in
    ``features/clinical_disclaimer.py``; the text itself rides in the page
    bootstrap. The same setting the desktop dialog writes, so acknowledging in
    either front-end settles it for both.
    """
    mark_shown(_settings())
    return {"acknowledged": True}


@app.get("/api/theme")
async def get_theme():
    return {"theme": _current_theme()}


@app.put("/api/theme")
async def set_theme(payload: ThemeUpdate):
    theme = _normalize_theme(payload.theme)
    settings = _settings()
    settings.set("theme", theme)
    return {"theme": theme}


@app.get("/api/preferences")
async def get_preferences():
    return _current_preferences()


@app.put("/api/preferences")
async def set_preferences(payload: PreferencesUpdate):
    model_size = payload.model_size if payload.model_size in SUPPORTED_MODELS else None
    if model_size is None:
        raise HTTPException(status_code=422, detail="Unsupported model size")

    accent = payload.accent if payload.accent in ACCENT_LABELS else "neutral"
    cleanup_level = payload.cleanup_level if payload.cleanup_level in CLEANUP_LEVELS else "medium"
    language = payload.language.strip() or "en"
    macro_region = payload.macro_region.strip()
    if macro_region not in macros.REGION_ORDER and macros.REGION_ORDER:
        macro_region = macros.REGION_ORDER[0]

    settings = _settings()
    settings.batch_set({
        "model_size": model_size,
        "language": language,
        "vad_filter": bool(payload.vad_filter),
        "accent": accent,
        "cleanup_level": cleanup_level,
        "last_macro_region": macro_region,
    })

    # Drop the cached engine so the next request builds the newly chosen model.
    with transcriber_lock:
        asr_engines.pop(resolve_model(model_size), None)

    return {
        "preferences": _current_preferences(),
        "macros": _macros_payload(),
    }


@app.get("/api/macros")
async def get_macros():
    return _macros_payload()


@app.post("/api/macros/reload")
async def reload_macros_route():
    try:
        reload_macros()
    except Exception as exc:
        logger.error("Failed to reload macros: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to reload macros") from exc
    return _macros_payload()


@app.get("/api/templates")
async def list_templates():
    names = _template_names()
    selected = _settings().get("last_template", "")
    if selected not in names:
        selected = names[0] if names else ""
    return {"templates": names, "selected": selected}


@app.post("/api/templates/{template_name}/load")
async def load_template(template_name: str):
    path = _resolve_template_path(template_name)
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.error("Failed to read template %s: %s", path, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to load template") from exc

    settings = _settings()
    settings.set("last_template", path.name)
    return {"name": path.name, "content": content}


#: The report currently open in the browser, and which of its findings have
#: been answered for. One instance, because this server is a loopback,
#: single-radiologist workstation with one report open at a time: the same
#: shape as the one desktop window. It is the *same* class the desktop reads
#: (``features/report_release.OutstandingFindings``), so the count, the marks
#: and the meaning of "acknowledged" cannot drift between the two front-ends.
#: Reset by ``/api/report/findings/reset`` when a new report starts.
_findings = OutstandingFindings()


@app.post("/api/report/findings")
async def report_findings_endpoint(payload: TextRequest):
    """Re-scan the open report and return what the gutter strip should show.

    Positions come back with the findings so the browser can place a mark
    without knowing any of the scanning rules: exactly what the desktop gutter
    reads off the same object.
    """
    _findings.update(payload.text)
    outstanding = {id(f) for f in _findings.outstanding}
    return {
        "count": _findings.count,
        "state": _findings.state,
        "findings": [
            {
                "term": f.term,
                "level": f.level,
                "start": f.start,
                "end": f.end,
                "outstanding": id(f) in outstanding,
            }
            for f in _findings.check.findings
        ],
    }


@app.post("/api/report/findings/reset")
async def reset_findings_endpoint():
    """Forget this report's findings and its acknowledgements: a new report.

    Acknowledgement answers "has this been phoned through for *this* patient",
    so carrying it into the next report would show the next patient's identical
    finding as already communicated.
    """
    _findings.reset()
    return {"count": 0, "state": "clear", "findings": []}


def _confirm_release(payload: ReportRequest) -> None:
    """Refuse to emit a report the radiologist has not answered for yet.

    The HTTP shape of the shared gate in ``features/report_release.py``: the rules
    and the audit trail live there, alongside the desktop front-end's dialog. Enforced
    here rather than in the browser so a client that forgets to ask cannot silently
    skip the warning.

    Both rules refuse with 409 and a ``reason``, one at a time. Unfilled fields go
    first because that is the only answer that can cancel: asking about findings
    first would audit an acknowledgement for a report that then did not leave.

    Raises:
        HTTPException: 409 with the unfilled field names while proceed_unfilled is
            False, then 409 with the findings while acknowledged is still None.
    """
    fields = unfilled_fields(payload.text)
    if fields and not payload.proceed_unfilled:
        raise HTTPException(
            status_code=409,
            detail={"reason": "unfilled_fields", "fields": list(fields)},
        )

    # Scanned through the shared state object, so this answer settles the same
    # count the strip in the browser is showing: and a finding typed in after
    # an acknowledgement puts it back to outstanding, rather than riding out on
    # an answer given before it existed.
    _findings.update(payload.text)
    check = _findings.check
    outstanding = _findings.outstanding
    if not check.needs_acknowledgement:
        return
    if payload.acknowledged is None:
        raise HTTPException(
            status_code=409,
            detail={
                "reason": "critical_findings",
                "summary": check.summary,
                "worst_level": check.worst_level,
            },
        )
    # Only what was still outstanding is audited: an audit entry claims a phone
    # call happened, so exporting the same report twice must not record two.
    record_release(
        outstanding,
        _model_to_dict(payload.patient).get("id", ""),
        payload.acknowledged,
    )
    if payload.acknowledged:
        _findings.acknowledge(check)


@app.post("/api/report/check")
async def check_report_endpoint(payload: ReportRequest):
    """Run the release gate without emitting a file.

    Copy puts the report on the clipboard, which leaves the app just as surely as a
    download does. It has no file to fetch, so it asks the gate this way instead.
    """
    _confirm_release(payload)
    return {"ok": True}


@app.post("/api/report/save-txt")
async def save_report_txt_endpoint(payload: ReportRequest):
    _confirm_release(payload)
    patient = _model_to_dict(payload.patient)
    content = format_plain_text_report(payload.text, patient)
    filename = _download_filename(patient, "txt")
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content=content, media_type="text/plain; charset=utf-8", headers=headers)


@app.post("/api/report/export-word")
async def export_word_endpoint(payload: ReportRequest):
    if not DOCX_AVAILABLE:
        raise HTTPException(status_code=503, detail="Word export is unavailable")

    _confirm_release(payload)

    patient = _model_to_dict(payload.patient)
    try:
        docx_bytes = export_to_word_bytes(payload.text, patient)
    except Exception as exc:
        logger.error("Word export failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to export Word document") from exc

    filename = _download_filename(patient, "docx")
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content=docx_bytes, media_type=DOCX_MIME, headers=headers)


@app.get("/api/terms/lookup")
async def term_lookup_endpoint(q: str = ""):
    """The two suggestion lists for a highlighted word or phrase.

    Thin: the ranking, the wordlists and the limit all live in
    ``src.medical.term_lookup``, so this front-end and the desktop one cannot
    show different neighbours for the same word. Runs off the event loop
    because the very first call builds the spelling index if start-up warming
    did not get there first.
    """
    limit = term_lookup.max_query_chars()
    if len(q) > limit:
        raise HTTPException(
            status_code=422, detail=f"Select at most {limit} characters"
        )
    result = await anyio.to_thread.run_sync(term_lookup.lookup, q)
    return asdict(result)


def _record_web_run(
    settings, prefs: dict, text: str, *, elapsed: float, audio_sec: float,
    stages: Optional[dict] = None, finalise_sec: float = 0.0, chunk_count: int = 0,
) -> None:
    """Write one run record for a browser dictation.

    Both browser paths land here. The one-shot upload has no separate "catching
    up after Stop", so it leaves *finalise_sec* at zero rather than inventing
    one; the live socket fills it in, because for streaming that number, how
    long after the last word the final text arrived, is the whole point.
    """
    record = run_log.start("web", model=str(prefs.get("model_size", "")))
    record.audio_sec = audio_sec
    record.duration_sec = round(elapsed, 3)
    record.finalise_sec = round(finalise_sec, 3)
    record.chunk_count = chunk_count
    record.word_count = len(text.split())
    record.stages = {n: {"total_ms": secs * 1000} for n, secs in (stages or {}).items()}
    if audio_sec > 0:
        record.decode_ratio = round(elapsed / audio_sec, 3)
    record.text = text if bool(settings.get("run_log_store_text", True)) else ""
    run_log.write(record, settings)
    # The one line that summarises the whole recording, so the console ends
    # with the verdict rather than leaving it to be added up by eye.
    event_log.emit(
        "run", "recording finished",
        audio_sec=round(audio_sec, 1), wall_sec=round(elapsed, 1),
        polish_sec=round(finalise_sec, 1), words=record.word_count,
        chunks=chunk_count,
        words_per_min=round(record.word_count / (audio_sec / 60), 1) if audio_sec > 1 else None,
        wall_per_audio_sec=record.decode_ratio,
    )


@app.post("/api/terms/suspect")
async def term_suspect_endpoint(payload: TextRequest):
    """Which words in the report are worth a second look, with their offsets.

    A POST because the body is the whole report, which does not belong in a
    query string. Same thinness as the lookup above: every rule about what
    counts as suspect lives in ``src.medical.term_lookup``, so the marks a
    radiologist sees in the browser are the marks they see on the desktop.
    """
    spans = await anyio.to_thread.run_sync(term_lookup.suspect_terms, payload.text)
    return {"spans": [asdict(span) for span in spans]}


@app.post("/api/terms/used")
async def term_used_endpoint():
    """Record that a suggestion was taken, so the marks hint can retire.

    Counted in settings rather than the browser because the desktop window
    counts the same takes into the same key: learning the feature in one
    front-end should not leave the other still explaining it.
    """
    settings = _settings()
    ceiling = int(settings.get("term_lookup_hint_uses", 3) or 3)
    used = int(settings.get("term_lookup_uses", 0) or 0)
    if used < ceiling:
        used += 1
        settings.set("term_lookup_uses", used)
    return {"term_lookup_uses": used, "term_lookup_hint_uses": ceiling}


@app.get("/api/runs")
async def runs_endpoint(limit: int = 100):
    """Every recorded dictation run, newest first (see features/run_log).

    Local diagnostics: this reads a file under ``data/`` and sends it to a page
    served on loopback. Nothing leaves the device.
    """
    limit = max(1, min(int(limit), 500))
    return {"runs": run_log.recent(limit=limit)}


@app.get("/api/debug/perf")
async def perf_endpoint():
    """Rolling stage timings for this process (local only, nothing is sent out).

    This is the "tracking" surface: it shows where dictation time actually goes
, per post-process stage, per transcription, so a slowdown can be pointed
    at rather than guessed at.
    """
    return {"stages": perf.snapshot(), "gauges": perf.gauges()}


@app.get("/api/debug/events")
async def events_endpoint(after: int = 0, limit: int = 500):
    """The live diary: what happened since event *after*, oldest first.

    The developer panel passes back the last sequence number it printed, so a
    poll only ever carries what is new. Local only: this reads a buffer in
    this process and serves it on loopback; nothing is stored or sent out.
    """
    return {
        "events": event_log.events(after=max(0, int(after)), limit=max(1, min(int(limit), 2000))),
        "latest": event_log.latest_seq(),
    }


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    settings = _settings()
    max_size = int(settings.get("max_upload_mb", get_default("max_upload_mb"))) * 1024 * 1024
    # Read one byte past the limit: enough to detect an oversized upload without
    # buffering the whole of it.
    audio_bytes = await file.read(max_size + 1)

    if len(audio_bytes) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {max_size / 1024 / 1024:.0f}MB"
        )

    file_like = io.BytesIO(audio_bytes)
    file_like.name = "audio.webm"

    try:
        t_start = time.time()
        prefs = _current_preferences()
        pause_threshold = settings.get("pause_threshold", 2.5)
        # A one-shot batch transcription of the whole clip, not the
        # latency-critical live loop: so it wants the same proper beam search as
        # the desktop's post-stop polish, and reads the same setting. Two names
        # for one decision is how the two front-ends drift apart.
        beam_size = int(settings.get("final_beam_size"))

        # Whisper transcription and post-processing are synchronous and
        # multi-second/CPU-bound. Run them in a worker thread so a request does
        # not block the event loop and stall every other concurrent request.
        # Timed per request as well as into perf. perf is never reset on this
        # path, so its snapshot is cumulative for the process: which is the
        # right thing for /api/debug/perf and the wrong thing for one run's
        # record. These are this run's own numbers.
        run_stages: dict = {}
        run_audio_sec = 0.0

        def _transcribe() -> str:
            nonlocal run_audio_sec
            t0 = time.time()
            with perf.stage("web.transcribe"):
                result = _get_engine().transcribe(
                    file_like,
                    TranscribeContext(
                        language=prefs["language"],
                        vad_filter=bool(prefs["vad_filter"]),
                        beam_size=beam_size,
                        condition_on_previous_text=False,
                        pause_threshold=pause_threshold,
                    ),
                )
                text = result.text
            run_stages["transcribe"] = round(time.time() - t0, 3)
            if result.segments:
                run_audio_sec = round(float(result.segments[-1].end), 3)
            t1 = time.time()
            with perf.stage("web.postprocess"):
                cleaned = postprocess_transcript(
                    text, accent=prefs["accent"], cleanup_level=prefs["cleanup_level"]
                )
            run_stages["postprocess"] = round(time.time() - t1, 3)
            return cleaned

        text = await anyio.to_thread.run_sync(_transcribe)
        elapsed = time.time() - t_start
        perf.record("web.request", elapsed)
        _record_web_run(settings, prefs, text, elapsed=elapsed,
                        audio_sec=run_audio_sec, stages=run_stages)
        return {"text": text}
    except Exception as e:
        logger.error("Transcription failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to transcribe audio"
        ) from e


# ---------------------------------------------------------------------------
# Live dictation over a WebSocket.
#
# The browser sends raw 16-bit PCM at 16 kHz: no container, no codec. That is
# the whole reason this can stream: webm/opus blobs from MediaRecorder are not
# independently decodable, so a live loop would have to re-decode the container
# from the start every cycle. Raw frames just append to a buffer.
#
# Down the wire: {"type": "partial"} while speaking, {"type": "final"} after
# Stop. Committed and preview text stay separate so the page can show a guess
# as a guess.
# ---------------------------------------------------------------------------



def _live_session(settings, prefs: dict) -> LiveSession:
    """Build a session from saved settings: the only place the knobs are read."""
    live_model = resolve_model(settings.get("live_model_size", get_default("live_model_size")))
    final_model = resolve_model(prefs.get("model_size") or settings.get("model_size"))
    # The knobs are read here and nowhere else, so this is the one honest place
    # to say which of them a recording actually ran with.
    event_log.emit(
        "live", "dictation session configured",
        live_model=live_model, final_model=final_model,
        language=prefs["language"], accent=prefs["accent"], cleanup=prefs["cleanup_level"],
        chunk_min_sec=float(settings.get("chunk_min_sec")),
        chunk_soft_max_sec=float(settings.get("chunk_soft_max_sec")),
        chunk_force_cut_sec=float(settings.get("chunk_force_cut_sec")),
        live_beam_size=int(settings.get("live_beam_size")),
        final_beam_size=int(settings.get("final_beam_size")),
        preview_max_lag_sec=float(settings.get("preview_max_lag_sec")),
        preview_min_tail_sec=float(
            settings.get("preview_min_tail_sec", get_default("preview_min_tail_sec"))
        ),
    )
    return LiveSession(
        _get_engine(live_model),
        _get_engine(final_model),
        language=prefs["language"],
        accent=prefs["accent"],
        cleanup_level=prefs["cleanup_level"],
        policy=ChunkPolicy(
            min_sec=float(settings.get("chunk_min_sec")),
            soft_max_sec=float(settings.get("chunk_soft_max_sec")),
            force_cut_sec=float(settings.get("chunk_force_cut_sec")),
            trailing_silence_sec=float(settings.get("chunk_trailing_silence_sec")),
        ),
        pause_threshold=float(settings.get("pause_threshold", 2.5)),
        live_beam_size=int(settings.get("live_beam_size")),
        final_beam_size=int(settings.get("final_beam_size")),
        silence_rms_floor=float(settings.get("silence_rms_floor")),
        silence_rms_margin=float(settings.get("silence_rms_margin")),
        preview_max_lag_sec=float(settings.get("preview_max_lag_sec")),
        polish_confidence_ceiling=float(settings.get("polish_confidence_ceiling")),
        uncertain_word_confidence=float(
            settings.get("uncertain_word_confidence", get_default("uncertain_word_confidence"))
        ),
        initial_prompt=build_context_prompt(),
        sr=LIVE_SAMPLE_RATE,
    )


@app.websocket("/ws/dictate")
async def dictate_socket(ws: WebSocket) -> None:
    await ws.accept()
    settings = _settings()
    prefs = _current_preferences()
    session = _live_session(settings, prefs)
    cycle_sec = float(settings.get("live_cycle_sec", get_default("live_cycle_sec")))
    max_sec = int(settings.get("max_upload_mb", get_default("max_upload_mb"))) * 1024 * 1024 / (2 * LIVE_SAMPLE_RATE)

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
                "committed": update.committed,
                "preview": update.preview,
                "state": update.state,
                "audioSec": round(update.audio_sec, 1),
                "uncertain": list(update.uncertain),
            })

    try:
        while True:
            message = await ws.receive()

            if message.get("type") == "websocket.disconnect":
                return

            if (raw := message.get("bytes")) is not None:
                # Reject an over-long recording at the same ceiling the upload
                # path uses, rather than letting one connection grow unbounded.
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
                command = json.loads(text).get("command")
            except (ValueError, AttributeError):
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
                    "text": session.committed_text(),
                    "uncertain": list(session.uncertain_words),
                })
                audio_ended = time.time()
                event_log.emit(
                    "live", "report handed back",
                    ms=(audio_ended - stop_at) * 1000,
                    words=len(session.committed_text().split()),
                )
                with event_log.timed("live", "accuracy pass", stage="web.finalize") as note:
                    text_out = await anyio.to_thread.run_sync(session.finalize)
                    note["words"] = len(text_out.split())
                    note["unsure_words"] = len(session.uncertain_words)
                await send({
                    "type": "final",
                    "text": text_out,
                    "uncertain": list(session.uncertain_words),
                })
                _record_web_run(
                    settings, prefs, text_out,
                    elapsed=time.time() - started,
                    audio_sec=session.audio_sec,
                    finalise_sec=time.time() - audio_ended,
                    chunk_count=session.chunks_decoded,
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
        await send({"type": "error", "message": "Dictation failed. Your audio is still in the browser: press Retry."})
    finally:
        if cycle_task is not None and not cycle_task.done():
            cycle_task.cancel()
        with contextlib.suppress(Exception):
            await ws.close()

if __name__ == "__main__":
    import uvicorn

    settings = _settings()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        app,
        host=str(settings.get("web_host", get_default("web_host"))),
        port=int(settings.get("web_port", get_default("web_port"))),
    )
