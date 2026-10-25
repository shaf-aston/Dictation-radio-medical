"""Settings the page changes: theme, preferences, templates, phrases, disclaimer."""

from __future__ import annotations

import logging
from pathlib import Path

import anyio
from fastapi import APIRouter, HTTPException

from src.core.paths import templates_dir
from src.dictation.asr import voice_model
from src.dictation.asr.factory import describe_engine
from src.dictation.asr.models import SUPPORTED_MODELS
from src.dictation.postprocess.pipeline import CLEANUP_LEVELS
from src.features import audit_log
from src.dictation.postprocess.accent import ACCENT_LABELS
from src.features.clinical_disclaimer import mark_shown
from src.medical import macros
from src.ui.web import state
from src.ui.web.pages import current_theme, normalize_theme
from src.ui.web.schemas import PreferencesUpdate, ThemeUpdate

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/disclaimer/ack")
async def acknowledge_disclaimer():
    """Record that the radiologist has read the first-launch disclaimer.

    The HTTP shape of the shared statement in
    ``features/clinical_disclaimer.py``; the text itself rides in the page
    bootstrap. The same setting the desktop dialog writes, so acknowledging in
    either front-end settles it for both.
    """
    mark_shown(state.settings())
    return {"acknowledged": True}


@router.get("/api/theme")
async def get_theme():
    return {"theme": current_theme()}


@router.put("/api/theme")
async def set_theme(payload: ThemeUpdate):
    theme = normalize_theme(payload.theme)
    state.settings().set("theme", theme)
    return {"theme": theme}


@router.get("/api/preferences")
async def get_preferences():
    return state.current_preferences()


@router.put("/api/preferences")
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
    settings = state.settings()
    voice_changed = payload.parakeet_model != settings.get("parakeet_model")
    if voice_changed:
        # Loads the new voice before answering (the page shows "Loading voice
        # model" meanwhile), so the next dictation does not pay for the load.
        try:
            await anyio.to_thread.run_sync(voice_model.switch, settings, payload.parakeet_model)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="Unknown voice model") from exc
        except Exception as exc:
            logger.error("Voice model %r failed to load: %s", payload.parakeet_model, exc, exc_info=True)
            raise HTTPException(
                status_code=503, detail="That voice model could not be loaded; the previous one is kept",
            ) from exc
    settings.batch_set({
        "model_size": model_size,
        "language": language,
        "vad_filter": bool(payload.vad_filter),
        "accent": accent,
        "cleanup_level": cleanup_level,
        "last_macro_region": macro_region,
    })
    # Drop the cached engines so the next request builds the newly chosen model.
    state.drop_engine(None if voice_changed else model_size)
    return {
        "preferences": state.current_preferences(),
        "macros": state.macros_payload(),
        # The badge names the engine that answers, which this save may have changed.
        "engine_label": describe_engine(settings),
    }


@router.post("/api/macros/reload")
async def reload_macros_route():
    try:
        macros.reload_macros()
    except Exception as exc:
        logger.error("Failed to reload macros: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to reload macros") from exc
    return {"macros": state.macros_payload()}


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


@router.post("/api/templates/{template_name}/load")
async def load_template(template_name: str):
    path = _resolve_template_path(template_name)
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.error("Failed to read template %s: %s", path, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to load template") from exc

    state.settings().set("last_template", path.name)
    audit_log.log_template_loaded(path.name)
    return {"name": path.name, "content": content}
