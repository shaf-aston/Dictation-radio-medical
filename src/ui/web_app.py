import html
import io
import json
import logging
import time
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Literal, Optional

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from src.core.settings import Settings
from src.dictation.postprocess.pipeline import postprocess_transcript
from src.dictation.transcriber import SUPPORTED_MODELS, Transcriber
from src.features.accent_corrections import ACCENT_LABELS
from src.features.file_manager import templates_dir
from src.features.report_manager import DOCX_AVAILABLE, export_to_word_bytes, format_plain_text_report
from src.medical import macros
from src.medical.macros import reload_macros

logger = logging.getLogger(__name__)

transcriber: Optional[Transcriber] = None
transcriber_model_size: Optional[str] = None
transcriber_lock = Lock()

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
    macro_region: str = ""


class PatientInfo(BaseModel):
    name: str = ""
    id: str = ""
    dob: str = ""
    study_date: str = ""
    referring: str = ""
    accession: str = ""


class ReportRequest(BaseModel):
    text: str = ""
    patient: PatientInfo = Field(default_factory=PatientInfo)


def _model_to_dict(model: BaseModel) -> dict:
    """Return model data across Pydantic versions."""
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


def _normalize_theme(theme: object) -> str:
    return str(theme) if str(theme) in {"dark", "light"} else "dark"


def _settings() -> Settings:
    return Settings()


def _current_theme() -> str:
    return _normalize_theme(_settings().get("theme", "dark"))


def _current_preferences() -> dict:
    settings = _settings()
    macro_region = settings.get("last_macro_region", macros.REGION_ORDER[0] if macros.REGION_ORDER else "")
    if macro_region not in macros.REGION_ORDER and macros.REGION_ORDER:
        macro_region = macros.REGION_ORDER[0]
    return {
        "model_size": settings.get("model_size", "base"),
        "language": settings.get("language", "en"),
        "vad_filter": settings.get("vad_filter", True),
        "accent": settings.get("accent", "neutral"),
        "macro_region": macro_region,
    }


def _current_patient_defaults() -> dict:
    return {
        "name": "",
        "id": "",
        "dob": "",
        "study_date": "",
        "referring": "",
        "accession": "",
    }


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


def _template_markup() -> tuple[str, str]:
    """Return option markup and disabled state for the template selector."""
    names = _template_names()
    if not names:
        return ('<option value="" disabled selected>No templates found</option>', "disabled")

    selected = _settings().get("last_template", "")
    if selected not in names:
        selected = names[0]

    options = []
    for name in names:
        selected_attr = " selected" if name == selected else ""
        options.append(
            f'<option value="{html.escape(name, quote=True)}"{selected_attr}>{html.escape(name)}</option>'
        )
    return "".join(options), ""


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
        "templates": templates,
        "selected_template": selected_template,
        "preferences": _current_preferences(),
        "patient_defaults": _current_patient_defaults(),
        "supported_models": SUPPORTED_MODELS,
        "accent_options": [
            {"key": key, "label": label}
            for key, label in ACCENT_LABELS.items()
        ],
        "macros": _macros_payload(),
        "docx_available": DOCX_AVAILABLE,
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
    template_options, template_disabled = _template_markup()
    bootstrap = _json_for_script(_bootstrap_payload())
    return (
        HTML_TEMPLATE
        .replace("__THEME__", _normalize_theme(theme))
        .replace("__THEME_STORAGE_KEY__", THEME_STORAGE_KEY)
        .replace("__TEMPLATE_OPTIONS__", template_options)
        .replace("__TEMPLATE_DISABLED__", template_disabled)
        .replace("__TEMPLATE_BUTTON_DISABLED__", template_disabled)
        .replace("__BOOTSTRAP__", bootstrap)
    )


def _get_transcriber() -> Transcriber:
    """Return a transcriber configured for the current saved model setting."""
    global transcriber, transcriber_model_size
    model_size = _settings().get("model_size", "base")
    if model_size not in SUPPORTED_MODELS:
        model_size = "base"

    with transcriber_lock:
        if transcriber is None or transcriber_model_size != model_size:
            transcriber = Transcriber(model_size=model_size)
            transcriber_model_size = model_size
        return transcriber


def _download_filename(patient: dict, ext: str) -> str:
    pid = (patient.get("id") or "unknown").strip().replace(" ", "_") or "unknown"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"dictation_{pid}_{ts}.{ext}"


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Prepare the app state on startup."""
    global transcriber, transcriber_model_size
    logger.info("Preparing web app state...")
    transcriber = None
    transcriber_model_size = None
    yield
    logger.info("Shutting down")


app = FastAPI(title="Radio Dictate Web", lifespan=lifespan)

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en" data-theme="__THEME__">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="color-scheme" content="dark light">
    <title>Radio Dictate - Web Workstation</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --bg: #f4f7fb;
            --bg-elevated: rgba(255, 255, 255, 0.82);
            --surface: #ffffff;
            --surface-soft: #eef4ff;
            --surface-muted: #f8fafc;
            --border: #d7e0ec;
            --text: #10213a;
            --text-muted: #5f6f86;
            --brand: #2563eb;
            --brand-hover: #1d4ed8;
            --brand-contrast: #ffffff;
            --danger: #dc2626;
            --danger-soft: #fee2e2;
            --success: #15803d;
            --shadow: 0 24px 60px rgba(15, 23, 42, 0.09);
            --focus: rgba(37, 99, 235, 0.28);
        }

        :root[data-theme="dark"] {
            --bg: #0b1020;
            --bg-elevated: rgba(12, 18, 33, 0.82);
            --surface: #111827;
            --surface-soft: #172033;
            --surface-muted: #0f172a;
            --border: #243044;
            --text: #e5e7eb;
            --text-muted: #94a3b8;
            --brand: #60a5fa;
            --brand-hover: #3b82f6;
            --brand-contrast: #0b1020;
            --danger: #f87171;
            --danger-soft: rgba(248, 113, 113, 0.16);
            --success: #4ade80;
            --shadow: 0 24px 70px rgba(2, 6, 23, 0.55);
            --focus: rgba(96, 165, 250, 0.32);
        }

        * {
            box-sizing: border-box;
        }

        html {
            color-scheme: light dark;
            background:
                radial-gradient(circle at top left, rgba(96, 165, 250, 0.18), transparent 30%),
                radial-gradient(circle at top right, rgba(14, 165, 233, 0.12), transparent 32%),
                var(--bg);
        }

        body {
            margin: 0;
            min-height: 100vh;
            color: var(--text);
            font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
            background: transparent;
        }

        .app-shell {
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        .topbar {
            position: sticky;
            top: 0;
            z-index: 20;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: 1rem 1.5rem;
            background: var(--bg-elevated);
            backdrop-filter: blur(16px);
            border-bottom: 1px solid var(--border);
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            min-width: 0;
        }

        .brand-mark {
            width: 2.75rem;
            height: 2.75rem;
            border-radius: 0.9rem;
            display: grid;
            place-items: center;
            color: var(--brand-contrast);
            background: linear-gradient(135deg, var(--brand), #0ea5e9);
            box-shadow: 0 10px 24px rgba(37, 99, 235, 0.24);
            flex: none;
        }

        .brand-text {
            display: flex;
            flex-direction: column;
            min-width: 0;
        }

        .brand-title {
            margin: 0;
            font-size: 1.05rem;
            font-weight: 700;
            line-height: 1.2;
        }

        .brand-subtitle {
            margin: 0.15rem 0 0;
            font-size: 0.88rem;
            color: var(--text-muted);
        }

        .topbar-actions {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            flex-wrap: wrap;
            justify-content: flex-end;
        }

        .btn {
            appearance: none;
            border: 1px solid transparent;
            border-radius: 0.9rem;
            padding: 0.72rem 1rem;
            font: inherit;
            font-weight: 600;
            cursor: pointer;
            transition: transform 160ms ease, background-color 160ms ease, border-color 160ms ease, color 160ms ease, opacity 160ms ease, box-shadow 160ms ease;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            min-height: 2.75rem;
        }

        .btn:hover {
            transform: translateY(-1px);
        }

        .btn:focus-visible,
        .textarea:focus-visible,
        .editor-shell:focus-within {
            outline: none;
            box-shadow: 0 0 0 4px var(--focus);
        }

        .btn:disabled {
            opacity: 0.62;
            cursor: progress;
            transform: none;
        }

        .btn-primary {
            background: var(--brand);
            color: var(--brand-contrast);
            box-shadow: 0 10px 24px rgba(37, 99, 235, 0.22);
        }

        .btn-primary:hover {
            background: var(--brand-hover);
        }

        .btn-ghost {
            background: transparent;
            color: var(--text-muted);
            border-color: var(--border);
        }

        .btn-ghost:hover {
            color: var(--text);
            background: var(--surface-muted);
        }

        .content {
            width: min(100%, 72rem);
            margin: 0 auto;
            padding: 1.5rem;
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .banner {
            display: flex;
            align-items: flex-start;
            gap: 0.75rem;
            padding: 1rem 1.1rem;
            border-radius: 1rem;
            background: var(--surface-soft);
            border: 1px solid var(--border);
            color: var(--text);
            box-shadow: var(--shadow);
        }

        .banner i {
            margin-top: 0.15rem;
            color: var(--brand);
        }

        .banner p {
            margin: 0;
            line-height: 1.5;
        }

        .report-card,
        .prefs-card,
        .macro-card {
            padding: 1rem 1.1rem;
            border-radius: 1rem;
            background: var(--surface);
            border: 1px solid var(--border);
            box-shadow: var(--shadow);
        }

        .card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            margin-bottom: 0.85rem;
        }

        .card-title {
            margin: 0;
            font-size: 1rem;
            font-weight: 700;
        }

        .card-subtitle {
            margin: 0.2rem 0 0;
            color: var(--text-muted);
            font-size: 0.9rem;
        }

        .field-grid {
            display: grid;
            gap: 0.75rem;
        }

        .field-grid.patient-grid {
            grid-template-columns: repeat(3, minmax(0, 1fr));
        }

        .field {
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
        }

        .field label {
            color: var(--text-muted);
            font-size: 0.88rem;
            font-weight: 600;
        }

        .field input,
        .field select {
            width: 100%;
            appearance: none;
            border: 1px solid var(--border);
            border-radius: 0.9rem;
            padding: 0.7rem 0.85rem;
            font: inherit;
            color: var(--text);
            background: var(--surface-muted);
            transition: box-shadow 160ms ease, border-color 160ms ease, background-color 160ms ease;
        }

        .field input:hover,
        .field select:hover {
            background: var(--surface-soft);
        }

        .field input:focus-visible,
        .field select:focus-visible {
            outline: none;
            box-shadow: 0 0 0 4px var(--focus);
        }

        .prefs-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
            align-items: end;
        }

        .prefs-row .field {
            min-width: 9rem;
            flex: 1 1 11rem;
        }

        .field.field--compact {
            flex: 0 1 8rem;
        }

        .check-field {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            min-height: 2.75rem;
            padding: 0.25rem 0.2rem 0;
            color: var(--text);
            font-weight: 600;
        }

        .check-field input {
            width: 1rem;
            height: 1rem;
            margin: 0;
        }

        /* ── Collapsible transcription settings ───────────────────────── */
        .settings-disclosure {
            border-radius: 1rem;
            background: var(--surface);
            border: 1px solid var(--border);
            box-shadow: var(--shadow);
        }

        .settings-disclosure > summary {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            padding: 0.9rem 1.1rem;
            cursor: pointer;
            list-style: none;
            user-select: none;
            border-radius: 1rem;
        }

        .settings-disclosure > summary::-webkit-details-marker { display: none; }

        .settings-disclosure > summary:hover { background: var(--surface-soft); }

        .settings-disclosure[open] > summary {
            border-radius: 1rem 1rem 0 0;
            border-bottom: 1px solid var(--border);
        }

        .settings-summary-title {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            font-size: 1rem;
            font-weight: 700;
        }

        .settings-badge-wrap {
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }

        .settings-badge {
            font-size: 0.82rem;
            color: var(--text-muted);
            font-family: monospace;
            transition: opacity 120ms ease;
        }

        .settings-disclosure[open] .settings-badge {
            opacity: 0;
            pointer-events: none;
        }

        .settings-chevron {
            color: var(--text-muted);
            font-size: 0.75rem;
            transition: transform 200ms ease;
            flex: none;
        }

        .settings-disclosure[open] .settings-chevron {
            transform: rotate(180deg);
        }

        .settings-body {
            padding: 0.85rem 1.1rem 1rem;
        }

        .settings-reload-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            margin-bottom: 0.75rem;
        }

        .macro-toolbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            flex-wrap: wrap;
            margin-bottom: 0.85rem;
        }

        .macro-region {
            min-width: 14rem;
            flex: 1 1 18rem;
        }

        .macro-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            max-height: 220px;
            overflow-y: auto;
        }

        .macro-chip {
            appearance: none;
            border: 1px solid var(--border);
            border-radius: 999px;
            padding: 0.6rem 0.85rem;
            background: var(--surface-muted);
            color: var(--text);
            cursor: pointer;
            transition: transform 160ms ease, background-color 160ms ease, box-shadow 160ms ease;
            display: inline-flex;
            flex-direction: column;
            align-items: flex-start;
            gap: 0.15rem;
            min-width: 9rem;
        }

        .macro-chip:hover {
            transform: translateY(-1px);
            background: var(--surface-soft);
        }

        .macro-chip strong {
            font-size: 0.92rem;
        }

        .macro-chip span {
            color: var(--text-muted);
            font-size: 0.8rem;
            line-height: 1.35;
            text-align: left;
        }

        .template-toolbar {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            flex-wrap: wrap;
            padding: 1rem 1.1rem;
            border-radius: 1rem;
            background: var(--surface);
            border: 1px solid var(--border);
            box-shadow: var(--shadow);
        }

        .template-toolbar label {
            font-weight: 600;
            color: var(--text-muted);
        }

        .template-select {
            flex: 1 1 18rem;
            min-width: 0;
            appearance: none;
            border: 1px solid var(--border);
            border-radius: 0.9rem;
            padding: 0.72rem 0.9rem;
            font: inherit;
            color: var(--text);
            background: var(--surface-muted);
            transition: box-shadow 160ms ease, border-color 160ms ease, background-color 160ms ease;
        }

        .template-select:hover {
            background: var(--surface-soft);
        }

        .template-select:disabled {
            opacity: 0.7;
            cursor: not-allowed;
        }

        .template-select:focus-visible {
            outline: none;
            box-shadow: 0 0 0 4px var(--focus);
        }

        .template-load {
            white-space: nowrap;
        }

        .editor-shell {
            position: relative;
            flex: 1;
            min-height: 28rem;
            border-radius: 1.4rem;
            background: var(--surface);
            border: 1px solid var(--border);
            box-shadow: var(--shadow);
            overflow: hidden;
            display: flex;
            transition: box-shadow 160ms ease, border-color 160ms ease;
        }

        .editor {
            width: 100%;
            min-height: 100%;
            border: 0;
            resize: none;
            padding: 1.4rem;
            color: var(--text);
            background: transparent;
            font-size: 1.05rem;
            line-height: 1.6;
            caret-color: var(--brand);
        }

        .editor::placeholder {
            color: var(--text-muted);
        }

        .status-chip {
            position: absolute;
            top: 1rem;
            right: 1rem;
            display: flex;
            align-items: center;
            gap: 0.55rem;
            padding: 0.7rem 0.85rem;
            border-radius: 999px;
            border: 1px solid var(--border);
            background: var(--bg-elevated);
            backdrop-filter: blur(16px);
            color: var(--text);
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
            opacity: 0;
            transition: opacity 220ms ease, transform 220ms ease;
            pointer-events: none;
        }

        .status-chip strong {
            font-weight: 600;
        }

        .status-chip.is-visible {
            opacity: 1;
            transform: translateY(0);
        }

        .status-chip.is-error {
            border-color: rgba(220, 38, 38, 0.35);
            color: var(--danger);
        }

        .status-dot {
            position: relative;
            width: 0.8rem;
            height: 0.8rem;
            border-radius: 999px;
            background: var(--dot-color, #ef4444);
            flex: none;
        }

        .status-dot::after {
            content: "";
            position: absolute;
            inset: -0.3rem;
            border-radius: inherit;
            background: var(--dot-glow, rgba(239, 68, 68, 0.22));
            animation: pulse 1.4s ease-in-out infinite;
        }

        .status-dot.is-processing {
            background: var(--dot-color, #3b82f6);
        }

        .status-dot.is-processing::after {
            background: var(--dot-glow, rgba(59, 130, 246, 0.22));
        }

        .controls {
            display: flex;
            justify-content: center;
            align-items: center;
            padding-top: 0.35rem;
            padding-bottom: 0.2rem;
        }

        .record-btn {
            width: 5.5rem;
            height: 5.5rem;
            border-radius: 999px;
            border: 0;
            display: grid;
            place-items: center;
            background: linear-gradient(135deg, var(--brand), #0ea5e9);
            color: var(--brand-contrast);
            box-shadow: 0 18px 36px rgba(37, 99, 235, 0.28);
            cursor: pointer;
            transition: transform 160ms ease, box-shadow 160ms ease, filter 160ms ease;
        }

        .record-btn:hover {
            transform: scale(1.03);
            filter: brightness(1.03);
        }

        .record-btn:active {
            transform: scale(0.97);
        }

        .record-btn:focus-visible {
            outline: none;
            box-shadow: 0 0 0 5px var(--focus), 0 18px 36px rgba(37, 99, 235, 0.28);
        }

        .record-btn.is-recording {
            background: linear-gradient(135deg, var(--danger), #fb7185);
            box-shadow: 0 18px 36px rgba(220, 38, 38, 0.28);
        }

        .record-btn.is-recording:focus-visible {
            box-shadow: 0 0 0 5px rgba(220, 38, 38, 0.2), 0 18px 36px rgba(220, 38, 38, 0.28);
        }

        .footnote {
            text-align: center;
            color: var(--text-muted);
            font-size: 0.92rem;
            padding-bottom: 0.35rem;
            margin: 0;
        }

        .kbd {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.15rem 0.45rem;
            border-radius: 0.45rem;
            border: 1px solid var(--border);
            background: var(--surface-muted);
            color: var(--text);
            font-size: 0.82rem;
            font-weight: 600;
        }

        .sr-only {
            position: absolute;
            width: 1px;
            height: 1px;
            padding: 0;
            margin: -1px;
            overflow: hidden;
            clip: rect(0, 0, 0, 0);
            white-space: nowrap;
            border: 0;
        }

        @keyframes pulse {
            0%, 100% { transform: scale(0.6); opacity: 0.5; }
            50% { transform: scale(1); opacity: 0.1; }
        }

        @media (max-width: 720px) {
            .topbar,
            .content {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .topbar {
                align-items: flex-start;
                flex-direction: column;
            }

            .topbar-actions {
                width: 100%;
                justify-content: space-between;
            }

            .brand-title {
                font-size: 1rem;
            }

            .field-grid.patient-grid {
                grid-template-columns: 1fr;
            }

            .prefs-row .field {
                min-width: 100%;
                flex-basis: 100%;
            }

            .macro-region {
                min-width: 100%;
            }

            .template-toolbar {
                align-items: stretch;
            }

            .template-toolbar label {
                width: 100%;
            }

            .template-select,
            .template-load {
                width: 100%;
                flex: 1 1 100%;
            }

            .editor {
                padding: 1rem;
                font-size: 1rem;
            }

            .status-chip {
                position: static;
                align-self: flex-end;
                margin: 0.9rem 0.9rem 0 0;
            }
        }
    </style>
</head>
<body>
    <div class="app-shell">
        <header class="topbar">
            <div class="brand">
                <div class="brand-mark" aria-hidden="true">
                    <i class="fas fa-stethoscope"></i>
                </div>
                <div class="brand-text">
                    <h1 class="brand-title">Radio Dictate Web</h1>
                    <p class="brand-subtitle">Private dictation with saved theme preference</p>
                </div>
            </div>

            <div class="topbar-actions">
                <button id="themeBtn" class="btn btn-ghost" type="button" aria-pressed="false" title="Toggle theme">
                    <i id="themeIcon" class="fas fa-moon"></i>
                    <span id="themeText">Dark</span>
                </button>
                <button id="newReportBtn" class="btn btn-ghost" type="button" title="Clear the editor and patient fields">
                    <i class="fas fa-file-circle-plus" aria-hidden="true"></i>
                    <span>New</span>
                </button>
                <button id="clearBtn" class="btn btn-ghost" type="button" title="Clear all text (cannot be undone)">
                    <i class="fas fa-trash"></i>
                    <span>Clear</span>
                </button>
                <button id="saveTxtBtn" class="btn btn-ghost" type="button" title="Download a plain-text report">
                    <i class="fas fa-file-arrow-down" aria-hidden="true"></i>
                    <span>Save TXT</span>
                </button>
                <button id="exportWordBtn" class="btn btn-ghost" type="button" title="Export a Word document">
                    <i class="fas fa-file-word" aria-hidden="true"></i>
                    <span>Word</span>
                </button>
                <button id="copyBtn" class="btn btn-primary" type="button" title="Copy text to clipboard">
                    <i id="copyIcon" class="fas fa-copy"></i>
                    <span id="copyText">Copy</span>
                </button>
            </div>
        </header>

        <main class="content">
            <section class="banner" aria-label="How to use">
                <i class="fas fa-circle-info" aria-hidden="true"></i>
                <p><strong>How to use:</strong> Click the microphone button to start recording. Your words will appear below.</p>
            </section>

            <section class="report-card" aria-label="Patient information">
                <div class="card-header">
                    <div>
                        <h2 class="card-title">Patient information</h2>
                        <p class="card-subtitle">Included in TXT and Word exports</p>
                    </div>
                </div>
                <div class="field-grid patient-grid">
                    <div class="field">
                        <label for="patientName">Name</label>
                        <input id="patientName" type="text" placeholder="Patient name" autocomplete="off">
                    </div>
                    <div class="field">
                        <label for="patientId">Patient ID</label>
                        <input id="patientId" type="text" placeholder="Patient ID" autocomplete="off">
                    </div>
                    <div class="field">
                        <label for="patientDob">DOB</label>
                        <input id="patientDob" type="text" placeholder="dd/mm/yyyy" autocomplete="off">
                    </div>
                    <div class="field">
                        <label for="patientStudyDate">Study date</label>
                        <input id="patientStudyDate" type="text" placeholder="Study date" autocomplete="off">
                    </div>
                    <div class="field">
                        <label for="patientReferrer">Referring clinician</label>
                        <input id="patientReferrer" type="text" placeholder="Referrer" autocomplete="off">
                    </div>
                    <div class="field">
                        <label for="patientAccession">Accession #</label>
                        <input id="patientAccession" type="text" placeholder="Accession" autocomplete="off">
                    </div>
                </div>
            </section>

            <details id="advancedSettings" class="settings-disclosure">
                <summary class="settings-summary">
                    <span class="settings-summary-title">
                        <i class="fas fa-sliders" aria-hidden="true"></i>
                        Transcription settings
                    </span>
                    <span class="settings-badge-wrap">
                        <span id="settingsBadge" class="settings-badge"></span>
                        <i class="fas fa-chevron-down settings-chevron" aria-hidden="true"></i>
                    </span>
                </summary>
                <div class="settings-body">
                    <div class="settings-reload-row">
                        <p class="card-subtitle">Model, language, VAD, and accent correction</p>
                        <button id="reloadMacrosBtn" class="btn btn-ghost" type="button" title="Reload quick phrases from macros.json">
                            <i class="fas fa-rotate-right" aria-hidden="true"></i>
                            <span>Reload macros</span>
                        </button>
                    </div>
                    <div class="prefs-row">
                        <div class="field field--compact">
                            <label for="modelSelect">Model</label>
                            <select id="modelSelect"></select>
                        </div>
                        <div class="field field--compact">
                            <label for="languageInput">Lang</label>
                            <input id="languageInput" type="text" maxlength="8" placeholder="en">
                        </div>
                        <div class="field field--compact">
                            <label for="accentSelect">Accent</label>
                            <select id="accentSelect"></select>
                        </div>
                        <label class="check-field" for="vadCheckbox">
                            <input id="vadCheckbox" type="checkbox">
                            <span>VAD</span>
                        </label>
                    </div>
                </div>
            </details>

            <section class="macro-card" aria-label="Quick phrases">
                <div class="macro-toolbar">
                    <div>
                        <h2 class="card-title">Quick phrases</h2>
                        <p class="card-subtitle">Tap a phrase to insert it at the cursor</p>
                    </div>
                    <div class="field macro-region">
                        <label for="macroRegionSelect">Region</label>
                        <select id="macroRegionSelect"></select>
                    </div>
                </div>
                <div id="macroButtons" class="macro-chips" aria-live="polite"></div>
            </section>

            <section class="template-toolbar" aria-label="Report templates">
                <label for="templateSelect">Template</label>
                <select id="templateSelect" class="template-select" __TEMPLATE_DISABLED__>
                    __TEMPLATE_OPTIONS__
                </select>
                <button id="loadTemplateBtn" class="btn btn-ghost template-load" type="button" __TEMPLATE_BUTTON_DISABLED__>
                    <i class="fas fa-file-import" aria-hidden="true"></i>
                    <span id="loadTemplateText">Load template</span>
                </button>
            </section>

            <section class="editor-shell" aria-label="Dictation editor">
                <div id="statusIndicator" class="status-chip" aria-live="polite" aria-label="Status indicator">
                    <span id="statusDot" class="status-dot" aria-hidden="true"></span>
                    <span id="statusText">Processing...</span>
                </div>
                <label for="editor" class="sr-only">Dictation content</label>
                <textarea id="editor" class="editor" placeholder="Your dictation will appear here..."></textarea>
            </section>

            <section class="controls" aria-label="Recording controls">
                <button id="dictateBtn" class="record-btn" type="button" aria-label="Start or stop recording" aria-pressed="false">
                    <i id="micIcon" class="fas fa-microphone" aria-hidden="true"></i>
                </button>
            </section>

            <p class="footnote"><span class="kbd">Space</span> to record &middot; <span class="kbd">Ctrl+Z</span> to undo</p>
        </main>
    </div>

    <script>
        const THEME_STORAGE_KEY = "__THEME_STORAGE_KEY__";
        const BOOTSTRAP = __BOOTSTRAP__;
        const PATIENT_STORAGE_KEY = "radio-dictate-web-patient";
        let mediaRecorder;
        let audioChunks = [];
        let isRecording = false;
        let undoStack = [''];
        let undoIndex = 0;
        let themeRequestInFlight = false;
        let templateLoadInFlight = false;
        let preferenceSaveInFlight = false;
        let preferenceSaveQueued = false;
        let preferenceSaveTimer = null;
        let macroReloadInFlight = false;
        let reportRequestInFlight = false;

        const dictateBtn = document.getElementById('dictateBtn');
        const micIcon = document.getElementById('micIcon');
        const editor = document.getElementById('editor');
        const newReportBtn = document.getElementById('newReportBtn');
        const saveTxtBtn = document.getElementById('saveTxtBtn');
        const exportWordBtn = document.getElementById('exportWordBtn');
        const templateSelect = document.getElementById('templateSelect');
        const loadTemplateBtn = document.getElementById('loadTemplateBtn');
        const loadTemplateText = document.getElementById('loadTemplateText');
        const patientName = document.getElementById('patientName');
        const patientId = document.getElementById('patientId');
        const patientDob = document.getElementById('patientDob');
        const patientStudyDate = document.getElementById('patientStudyDate');
        const patientReferrer = document.getElementById('patientReferrer');
        const patientAccession = document.getElementById('patientAccession');
        const modelSelect = document.getElementById('modelSelect');
        const languageInput = document.getElementById('languageInput');
        const accentSelect = document.getElementById('accentSelect');
        const vadCheckbox = document.getElementById('vadCheckbox');
        const reloadMacrosBtn = document.getElementById('reloadMacrosBtn');
        const macroRegionSelect = document.getElementById('macroRegionSelect');
        const macroButtons = document.getElementById('macroButtons');
        const clearBtn = document.getElementById('clearBtn');
        const copyBtn = document.getElementById('copyBtn');
        const copyText = document.getElementById('copyText');
        const copyIcon = document.getElementById('copyIcon');
        const statusIndicator = document.getElementById('statusIndicator');
        const statusText = document.getElementById('statusText');
        const statusDot = document.getElementById('statusDot');
        const themeBtn = document.getElementById('themeBtn');
        const themeText = document.getElementById('themeText');
        const themeIcon = document.getElementById('themeIcon');

        function normalizeTheme(value) {
            return value === 'light' ? 'light' : 'dark';
        }

        function currentTheme() {
            return normalizeTheme(document.documentElement.dataset.theme || 'dark');
        }

        function renderTheme(theme, persist = true) {
            const next = normalizeTheme(theme);
            document.documentElement.dataset.theme = next;
            document.documentElement.style.colorScheme = next;
            themeBtn.setAttribute('aria-pressed', next === 'dark' ? 'true' : 'false');
            themeBtn.title = next === 'dark' ? 'Switch to light mode' : 'Switch to dark mode';
            themeText.textContent = next === 'dark' ? 'Dark' : 'Light';
            themeIcon.className = next === 'dark' ? 'fas fa-moon' : 'fas fa-sun';
            if (persist) {
                localStorage.setItem(THEME_STORAGE_KEY, next);
            }
        }

        function setThemeSavingState(isSaving) {
            themeBtn.disabled = isSaving;
            if (isSaving) {
                themeText.textContent = 'Saving...';
            } else {
                renderTheme(currentTheme(), true);
            }
        }

        function showStatus(message, kind = 'info', timeout = 0) {
            statusIndicator.classList.add('is-visible');
            statusIndicator.classList.toggle('is-error', kind === 'error');
            statusText.textContent = message;
            statusDot.classList.toggle('is-processing', kind === 'processing');
            if (kind === 'processing') {
                statusDot.style.setProperty('--dot-color', 'var(--brand)');
                statusDot.style.setProperty('--dot-glow', 'rgba(96, 165, 250, 0.22)');
            } else if (kind === 'success') {
                statusDot.style.setProperty('--dot-color', 'var(--success)');
                statusDot.style.setProperty('--dot-glow', 'rgba(74, 222, 128, 0.22)');
            } else if (kind === 'error') {
                statusDot.style.setProperty('--dot-color', 'var(--danger)');
                statusDot.style.setProperty('--dot-glow', 'rgba(248, 113, 113, 0.22)');
            } else {
                statusDot.style.setProperty('--dot-color', 'var(--brand)');
                statusDot.style.setProperty('--dot-glow', 'rgba(96, 165, 250, 0.22)');
            }
            if (timeout) {
                setTimeout(() => {
                    statusIndicator.classList.remove('is-visible', 'is-error');
                }, timeout);
            }
        }

        function hideStatus() {
            statusIndicator.classList.remove('is-visible', 'is-error');
            statusDot.style.removeProperty('--dot-color');
            statusDot.style.removeProperty('--dot-glow');
        }

        function showError(message) {
            showStatus(message, 'error', 5000);
            statusText.style.color = '';
        }

        async function persistTheme(theme) {
            if (themeRequestInFlight) {
                return;
            }
            themeRequestInFlight = true;
            const previousTheme = currentTheme();
            setThemeSavingState(true);
            renderTheme(theme, false);

            try {
                const response = await fetch('/api/theme', {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ theme })
                });

                const result = await response.json().catch(() => ({}));
                if (!response.ok) {
                    throw new Error(result.detail || 'Theme update failed');
                }

                renderTheme(normalizeTheme(result.theme || theme), true);
            } catch (err) {
                renderTheme(previousTheme, true);
                showError('Theme preference could not be saved: ' + err.message);
            } finally {
                setThemeSavingState(false);
                themeRequestInFlight = false;
            }
        }

        function toggleTheme() {
            persistTheme(currentTheme() === 'dark' ? 'light' : 'dark');
        }

        function syncCachedTheme() {
            const savedTheme = localStorage.getItem(THEME_STORAGE_KEY);
            const serverTheme = currentTheme();
            if (savedTheme && normalizeTheme(savedTheme) !== serverTheme) {
                localStorage.setItem(THEME_STORAGE_KEY, serverTheme);
            } else if (!savedTheme) {
                localStorage.setItem(THEME_STORAGE_KEY, serverTheme);
            }
        }

        function safeParseJson(raw, fallback) {
            try {
                return raw ? JSON.parse(raw) : fallback;
            } catch (err) {
                return fallback;
            }
        }

        function loadPatientDraft() {
            return safeParseJson(localStorage.getItem(PATIENT_STORAGE_KEY), BOOTSTRAP.patient_defaults || {});
        }

        function savePatientDraft() {
            localStorage.setItem(PATIENT_STORAGE_KEY, JSON.stringify(collectPatientInfo()));
        }

        function collectPatientInfo() {
            return {
                name: patientName.value.trim(),
                id: patientId.value.trim(),
                dob: patientDob.value.trim(),
                study_date: patientStudyDate.value.trim(),
                referring: patientReferrer.value.trim(),
                accession: patientAccession.value.trim(),
            };
        }

        function applyPatientInfo(info) {
            const data = Object.assign({}, BOOTSTRAP.patient_defaults || {}, info || {});
            patientName.value = data.name || '';
            patientId.value = data.id || '';
            patientDob.value = data.dob || '';
            patientStudyDate.value = data.study_date || '';
            patientReferrer.value = data.referring || '';
            patientAccession.value = data.accession || '';
        }

        function populateModelOptions() {
            modelSelect.innerHTML = '';
            (BOOTSTRAP.supported_models || []).forEach((model) => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                modelSelect.appendChild(option);
            });
        }

        function populateAccentOptions() {
            accentSelect.innerHTML = '';
            (BOOTSTRAP.accent_options || []).forEach((accent) => {
                const option = document.createElement('option');
                option.value = accent.key;
                option.textContent = accent.label;
                accentSelect.appendChild(option);
            });
        }

        function populateMacroRegions() {
            macroRegionSelect.innerHTML = '';
            (BOOTSTRAP.macros?.regions || []).forEach((region) => {
                const option = document.createElement('option');
                option.value = region.name;
                option.textContent = region.name;
                macroRegionSelect.appendChild(option);
            });
        }

        function currentMacroRegion() {
            return macroRegionSelect.value || (BOOTSTRAP.macros?.selected_region || '');
        }

        function renderMacros(regionName) {
            const region = (BOOTSTRAP.macros?.regions || []).find((item) => item.name === regionName) || BOOTSTRAP.macros?.regions?.[0];
            macroButtons.innerHTML = '';

            if (!region || !region.phrases || region.phrases.length === 0) {
                const empty = document.createElement('div');
                empty.className = 'card-subtitle';
                empty.textContent = 'No quick phrases are available in this region.';
                macroButtons.appendChild(empty);
                return;
            }

            region.phrases.forEach((phrase) => {
                const btn = document.createElement('button');
                btn.type = 'button';
                btn.className = 'macro-chip';

                const title = document.createElement('strong');
                title.textContent = phrase.label;
                btn.appendChild(title);

                const preview = document.createElement('span');
                preview.textContent = phrase.text.length > 78 ? `${phrase.text.slice(0, 75)}...` : phrase.text;
                btn.appendChild(preview);

                btn.addEventListener('click', () => insertTextAtCursor(phrase.text));
                macroButtons.appendChild(btn);
            });
        }

        function pushUndoState() {
            if (undoIndex < undoStack.length - 1) {
                undoStack.length = undoIndex + 1;
            }
            undoStack.push(editor.value);
            undoIndex++;
            if (undoStack.length > 50) {
                undoStack.shift();
                undoIndex = undoStack.length - 1;
            }
        }

        function setEditorValue(value, { moveCaretToEnd = true } = {}) {
            editor.value = value;
            if (moveCaretToEnd && typeof editor.setSelectionRange === 'function') {
                const end = editor.value.length;
                editor.setSelectionRange(end, end);
            }
            pushUndoState();
        }

        function insertTextAtCursor(text) {
            const current = editor.value;
            const start = typeof editor.selectionStart === 'number' ? editor.selectionStart : current.length;
            const end = typeof editor.selectionEnd === 'number' ? editor.selectionEnd : current.length;
            const before = current.slice(0, start);
            const after = current.slice(end);
            const needsSpacer = before.length > 0 && !/[\s\n]$/.test(before) && !/^[\s\n]/.test(text);
            const insertion = `${needsSpacer ? ' ' : ''}${text}`;
            editor.value = before + insertion + after;
            if (typeof editor.setSelectionRange === 'function') {
                const pos = before.length + insertion.length;
                editor.setSelectionRange(pos, pos);
            }
            editor.focus();
            pushUndoState();
        }

        function loadReportSettingsFromServer() {
            const prefs = BOOTSTRAP.preferences || {};
            modelSelect.value = prefs.model_size || (BOOTSTRAP.supported_models || [])[0] || 'base';
            languageInput.value = prefs.language || 'en';
            accentSelect.value = prefs.accent || 'neutral';
            vadCheckbox.checked = Boolean(prefs.vad_filter);
            macroRegionSelect.value = prefs.macro_region || currentMacroRegion();
            if (!macroRegionSelect.value && macroRegionSelect.options.length > 0) {
                macroRegionSelect.selectedIndex = 0;
            }
            templateSelect.value = BOOTSTRAP.selected_template || templateSelect.value;
            BOOTSTRAP.macros = BOOTSTRAP.macros || { regions: [], selected_region: '' };
            renderMacros(currentMacroRegion());
            syncTemplateButton();
            syncReportButtons();
            updateSettingsBadge();
        }

        function updateSettingsBadge() {
            const badge = document.getElementById('settingsBadge');
            if (!badge) return;
            const m = modelSelect.value || 'base';
            const l = languageInput.value.trim() || 'en';
            const a = accentSelect.options[accentSelect.selectedIndex]?.text || 'Neutral';
            const v = vadCheckbox.checked ? 'VAD' : '';
            badge.textContent = [m, l, a, v].filter(Boolean).join(' · ');
        }

        function applyBootstrapState() {
            populateModelOptions();
            populateAccentOptions();
            populateMacroRegions();
            loadPatientFromStorage();
            loadReportSettingsFromServer();
            BOOTSTRAP.docx_available = Boolean(BOOTSTRAP.docx_available);
            exportWordBtn.disabled = !BOOTSTRAP.docx_available;
        }

        function syncReportButtons() {
            exportWordBtn.disabled = !BOOTSTRAP.docx_available || reportRequestInFlight;
            exportWordBtn.title = BOOTSTRAP.docx_available
                ? 'Export a Word document'
                : 'Install python-docx to enable Word export';
            saveTxtBtn.disabled = reportRequestInFlight;
            newReportBtn.disabled = reportRequestInFlight;
            clearBtn.disabled = reportRequestInFlight;
            copyBtn.disabled = reportRequestInFlight;
        }

        function schedulePreferenceSave() {
            if (preferenceSaveTimer) {
                clearTimeout(preferenceSaveTimer);
            }
            preferenceSaveTimer = setTimeout(() => {
                persistPreferences();
            }, 250);
        }

        async function persistPreferences() {
            if (preferenceSaveInFlight) {
                preferenceSaveQueued = true;
                return;
            }
            preferenceSaveInFlight = true;
            const payload = {
                model_size: modelSelect.value || 'base',
                language: languageInput.value.trim() || 'en',
                vad_filter: vadCheckbox.checked,
                accent: accentSelect.value || 'neutral',
                macro_region: macroRegionSelect.value || '',
            };

            try {
                const response = await fetch('/api/preferences', {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                const result = await response.json().catch(() => ({}));
                if (!response.ok) {
                    throw new Error(result.detail || 'Preference update failed');
                }

                BOOTSTRAP.preferences = Object.assign({}, payload, result.preferences || {});
                if (result.macros) {
                    BOOTSTRAP.macros = result.macros;
                }
                modelSelect.value = BOOTSTRAP.preferences.model_size || payload.model_size;
                languageInput.value = BOOTSTRAP.preferences.language || payload.language;
                accentSelect.value = BOOTSTRAP.preferences.accent || payload.accent;
                vadCheckbox.checked = Boolean(BOOTSTRAP.preferences.vad_filter);
                macroRegionSelect.value = BOOTSTRAP.preferences.macro_region || payload.macro_region;
                renderMacros(currentMacroRegion());
                showStatus('Settings saved', 'success', 1200);
            } catch (err) {
                showError('Settings could not be saved: ' + err.message);
            } finally {
                preferenceSaveInFlight = false;
                syncReportButtons();
                if (preferenceSaveQueued) {
                    preferenceSaveQueued = false;
                    persistPreferences();
                }
            }
        }

        async function reloadMacros() {
            if (macroReloadInFlight) {
                return;
            }
            macroReloadInFlight = true;
            reloadMacrosBtn.disabled = true;
            reloadMacrosBtn.innerHTML = '<i class="fas fa-spinner fa-spin" aria-hidden="true"></i><span>Reloading...</span>';
            try {
                const response = await fetch('/api/macros/reload', { method: 'POST' });
                const result = await response.json().catch(() => ({}));
                if (!response.ok) {
                    throw new Error(result.detail || 'Macro reload failed');
                }
                BOOTSTRAP.macros = result.macros || BOOTSTRAP.macros;
                populateMacroRegions();
                macroRegionSelect.value = result.selected_region || macroRegionSelect.value;
                renderMacros(currentMacroRegion());
                showStatus('Macros reloaded', 'success', 1500);
            } catch (err) {
                showError('Macros could not be reloaded: ' + err.message);
            } finally {
                macroReloadInFlight = false;
                reloadMacrosBtn.disabled = false;
                reloadMacrosBtn.innerHTML = '<i class="fas fa-rotate-right" aria-hidden="true"></i><span>Reload macros</span>';
                syncReportButtons();
            }
        }

        function loadPatientFromStorage() {
            applyPatientInfo(loadPatientDraft());
        }

        function resetPatientInfo() {
            applyPatientInfo(BOOTSTRAP.patient_defaults || {});
            savePatientDraft();
        }

        function confirmTemplateFieldCompletion() {
            const matches = editor.value.match(/\[([A-Z][A-Z0-9 _/-]{1,40})\]|\{\{([^}]{1,40})\}\}/g);
            if (!matches || matches.length === 0) {
                return true;
            }
            const unique = Array.from(new Set(matches));
            return confirm(
                `The report contains ${unique.length} unfilled field(s):\n\n` +
                unique.slice(0, 10).map((item) => `• ${item}`).join('\n') +
                (unique.length > 10 ? '\n…' : '') +
                '\n\nContinue anyway?'
            );
        }

        function parseDownloadFilename(contentDisposition, fallback) {
            const match = /filename\*=UTF-8''([^;]+)|filename="?([^";]+)"?/i.exec(contentDisposition || '');
            const candidate = match && (match[1] || match[2]);
            return candidate ? decodeURIComponent(candidate) : fallback;
        }

        function triggerDownload(blob, filename) {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            a.remove();
            setTimeout(() => URL.revokeObjectURL(url), 1000);
        }

        async function downloadReport(kind) {
            if (reportRequestInFlight) {
                return;
            }
            if (!confirmTemplateFieldCompletion()) {
                return;
            }

            reportRequestInFlight = true;
            syncReportButtons();
            const endpoint = kind === 'word' ? '/api/report/export-word' : '/api/report/save-txt';
            const fallbackName = kind === 'word' ? 'radiology_report.docx' : 'radiology_report.txt';

            try {
                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        text: editor.value,
                        patient: collectPatientInfo(),
                    }),
                });
                if (!response.ok) {
                    const errorText = await response.text();
                    let detail = 'Report download failed';
                    try {
                        const parsed = JSON.parse(errorText);
                        detail = parsed.detail || detail;
                    } catch (err) {
                        if (errorText) {
                            detail = errorText;
                        }
                    }
                    throw new Error(detail);
                }

                const blob = await response.blob();
                const filename = parseDownloadFilename(response.headers.get('content-disposition'), fallbackName);
                triggerDownload(blob, filename);
                showStatus(kind === 'word' ? 'Word export ready' : 'TXT download ready', 'success', 1800);
            } catch (err) {
                showError('Report export failed: ' + err.message);
            } finally {
                reportRequestInFlight = false;
                syncReportButtons();
            }
        }

        function syncTemplateButton() {
            loadTemplateBtn.disabled = templateLoadInFlight || !templateSelect.value;
            loadTemplateBtn.title = templateSelect.value
                ? `Load ${templateSelect.value}`
                : 'Select a template first';
        }

        function setTemplateLoadingState(isLoading) {
            templateLoadInFlight = isLoading;
            loadTemplateText.textContent = isLoading ? 'Loading...' : 'Load template';
            syncTemplateButton();
        }

        async function loadSelectedTemplate() {
            const name = templateSelect.value;
            if (!name || templateLoadInFlight) {
                return;
            }

            try {
                setTemplateLoadingState(true);
                showStatus(`Loading template: ${name}`, 'processing');

                const response = await fetch(`/api/templates/${encodeURIComponent(name)}/load`, {
                    method: 'POST',
                });
                const result = await response.json().catch(() => ({}));
                if (!response.ok) {
                    throw new Error(result.detail || 'Template load failed');
                }

                editor.value = result.content || '';
                if (typeof editor.setSelectionRange === 'function') {
                    const end = editor.value.length;
                    editor.setSelectionRange(end, end);
                }
                editor.focus();
                undoStack = [editor.value];
                undoIndex = 0;
                showStatus(`Template loaded: ${result.name}`, 'success', 2000);
            } catch (err) {
                showError('Template load failed: ' + err.message);
            } finally {
                setTemplateLoadingState(false);
            }
        }

        // Track text changes for undo and persist patient fields locally.
        editor.addEventListener('input', pushUndoState);
        [patientName, patientId, patientDob, patientStudyDate, patientReferrer, patientAccession].forEach((field) => {
            field.addEventListener('input', () => {
                savePatientDraft();
            });
        });

        newReportBtn.addEventListener('click', () => {
            const hasContent = editor.value.trim() !== '' || Object.values(collectPatientInfo()).some(Boolean);
            if (hasContent && !confirm('Clear the current report and patient information?')) {
                return;
            }
            editor.value = '';
            undoStack = [''];
            undoIndex = 0;
            editor.focus();
            resetPatientInfo();
            hideStatus();
        });

        // Clear with confirmation
        clearBtn.addEventListener('click', () => {
            if (editor.value.trim() === '') return;
            if (confirm('Are you sure you want to clear all text? This cannot be undone.')) {
                setEditorValue('');
                hideStatus();
            }
        });

        saveTxtBtn.addEventListener('click', () => downloadReport('txt'));
        exportWordBtn.addEventListener('click', () => downloadReport('word'));

        // Copy with visual feedback
        copyBtn.addEventListener('click', () => {
            if (editor.value.trim() === '') {
                copyText.textContent = 'Nothing to copy';
                setTimeout(() => { copyText.textContent = 'Copy'; }, 2000);
                return;
            }
            navigator.clipboard.writeText(editor.value)
                .then(() => {
                    copyText.textContent = 'Copied!';
                    copyIcon.className = 'fas fa-check';
                    setTimeout(() => {
                        copyText.textContent = 'Copy';
                        copyIcon.className = 'fas fa-copy';
                    }, 2000);
                })
                .catch(err => {
                    console.error('Failed to copy', err);
                    copyText.textContent = 'Copy failed';
                    setTimeout(() => { copyText.textContent = 'Copy'; }, 2000);
                });
        });

        modelSelect.addEventListener('change', () => { schedulePreferenceSave(); updateSettingsBadge(); });
        languageInput.addEventListener('input', () => { schedulePreferenceSave(); updateSettingsBadge(); });
        accentSelect.addEventListener('change', () => { schedulePreferenceSave(); updateSettingsBadge(); });
        vadCheckbox.addEventListener('change', () => { schedulePreferenceSave(); updateSettingsBadge(); });
        macroRegionSelect.addEventListener('change', () => {
            renderMacros(currentMacroRegion());
            schedulePreferenceSave();
        });
        reloadMacrosBtn.addEventListener('click', reloadMacros);
        templateSelect.addEventListener('change', syncTemplateButton);
        loadTemplateBtn.addEventListener('click', loadSelectedTemplate);
        themeBtn.addEventListener('click', toggleTheme);

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                if (e.key === 'z' && !isRecording) { e.preventDefault(); undo(); }
                if (e.key === 'y' && !isRecording) { e.preventDefault(); redo(); }
                if (e.key === 'n') { e.preventDefault(); newReportBtn.click(); }
                if (e.key === 's') { e.preventDefault(); saveTxtBtn.click(); }
                if (e.shiftKey && e.key.toLowerCase() === 'w') { e.preventDefault(); exportWordBtn.click(); }
                if (e.key === 't') { e.preventDefault(); loadTemplateBtn.click(); }
                if (e.key === 'r') { e.preventDefault(); reloadMacrosBtn.click(); }
            }
            if (e.code === 'Space' && e.target === document.body && !isRecording) {
                e.preventDefault();
                dictateBtn.click();
            }
        });

        function undo() {
            if (undoIndex > 0) {
                undoIndex--;
                editor.value = undoStack[undoIndex];
                editor.focus();
            }
        }

        function redo() {
            if (undoIndex < undoStack.length - 1) {
                undoIndex++;
                editor.value = undoStack[undoIndex];
                editor.focus();
            }
        }

        async function startRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);

                mediaRecorder.ondataavailable = event => {
                    if (event.data.size > 0) audioChunks.push(event.data);
                };

                mediaRecorder.onstop = sendAudio;

                audioChunks = [];
                mediaRecorder.start();
                isRecording = true;
                dictateBtn.setAttribute('aria-pressed', 'true');
                dictateBtn.classList.add('is-recording');
                micIcon.className = 'fas fa-stop';

                showStatus('Recording...', 'processing');
            } catch (err) {
                console.error('Microphone access denied:', err);
                showError('Microphone access denied. Please allow microphone permissions in your browser settings and try again.');
            }
        }

        function stopRecording() {
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
                isRecording = false;
                dictateBtn.setAttribute('aria-pressed', 'false');
                dictateBtn.classList.remove('is-recording');
                micIcon.className = 'fas fa-microphone';

                showStatus('Transcribing...', 'processing');
            }
        }

        dictateBtn.addEventListener('click', () => {
            if (isRecording) {
                stopRecording();
            } else {
                startRecording();
            }
        });

        async function sendAudio() {
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            const formData = new FormData();
            formData.append("file", audioBlob, "dictation.webm");

            try {
                const response = await fetch('/transcribe', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Transcription failed');
                }

                const result = await response.json();

                if (result.text) {
                    const nextValue = editor.value.trim() !== '' ? `${editor.value} ${result.text}` : result.text;
                    setEditorValue(nextValue);
                    editor.scrollTop = editor.scrollHeight;
                    showStatus('Transcription complete', 'success', 2000);
                } else {
                    hideStatus();
                }
            } catch (err) {
                console.error('Transcription error:', err);
                showError('Transcription failed: ' + err.message + '. Try again.');
            } finally {
                statusDot.classList.remove('is-processing');
            }
        }

        dictateBtn.setAttribute('aria-pressed', 'false');
        dictateBtn.setAttribute('role', 'button');
        dictateBtn.setAttribute('aria-label', 'Start or stop recording');
        renderTheme(document.documentElement.dataset.theme || 'dark', false);
        syncCachedTheme();
        applyBootstrapState();
        syncReportButtons();
    </script>
</body>
</html>
"""


@app.get("/")
async def root():
    return HTMLResponse(_render_html(_current_theme()))


@app.get("/favicon.ico")
async def favicon():
    favicon_path = Path(__file__).parent / "frontends" / "favicon.svg"
    if favicon_path.exists():
        return Response(content=favicon_path.read_bytes(), media_type="image/svg+xml")
    return Response(content=b"", status_code=204)


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
        "last_macro_region": macro_region,
    })

    # Drop any cached transcriber so the next request picks up the new model.
    global transcriber, transcriber_model_size
    with transcriber_lock:
        if transcriber_model_size != model_size:
            transcriber = None
            transcriber_model_size = None

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


@app.post("/api/report/save-txt")
async def save_report_txt_endpoint(payload: ReportRequest):
    patient = _model_to_dict(payload.patient)
    content = format_plain_text_report(payload.text, patient)
    filename = _download_filename(patient, "txt")
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content=content, media_type="text/plain; charset=utf-8", headers=headers)


@app.post("/api/report/export-word")
async def export_word_endpoint(payload: ReportRequest):
    if not DOCX_AVAILABLE:
        raise HTTPException(status_code=503, detail="Word export is unavailable")

    patient = _model_to_dict(payload.patient)
    try:
        docx_bytes = export_to_word_bytes(payload.text, patient)
    except Exception as exc:
        logger.error("Word export failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to export Word document") from exc

    filename = _download_filename(patient, "docx")
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content=docx_bytes, media_type=DOCX_MIME, headers=headers)


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # Read uploaded bytes with size limit (50MB max)
    max_size = 50 * 1024 * 1024  # 50MB
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
        t_transcribe_start = time.time()
        text, _ = _get_transcriber().transcribe(
            file_like,
            language=prefs["language"],
            vad_filter=bool(prefs["vad_filter"]),
            beam_size=1,
            condition_on_previous_text=False,
            pause_threshold=_settings().get("pause_threshold", 2.5),
        )
        transcribe_time = time.time() - t_transcribe_start
        text = postprocess_transcript(text, accent=prefs["accent"])
        total_time = time.time() - t_start
        logger.info("Request [%.2fs] (transcribe: [%.2fs], post-process: [%.2fs])",
                    total_time, transcribe_time, total_time - transcribe_time)
        return {"text": text}
    except Exception as e:
        logger.error("Transcription failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to transcribe audio"
        ) from e


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="127.0.0.1", port=8005)
