# CLAUDE.md — Radio Dictate

Radio Dictate is a local, offline medical dictation workstation for radiologists.
Speech → Whisper (local, no cloud) → radiology corrections pipeline → formatted report.
Primary users: MSK radiologists. Templates cover chest, neuro, abdominal, and ultrasound too.

---

## Running

```bash
python -m src.ui.app          # Desktop GUI (PySide6)
python -m src.ui.web_app      # Web interface (FastAPI)
python scripts/verify_setup.py
```

Always run from the **project root**. `src.*` imports break if you run from inside `src/`.

---

## Architecture

```
src/
├── dictation/                The full speech→corrected-text pipeline
│   ├── audio.py              Recorder (sounddevice → WAV)
│   ├── transcriber.py        faster-whisper wrapper + segment hallucination filter
│   ├── worker.py             LiveTranscribeWorker (QThread, sliding window)
│   ├── text_diff.py          boundary deduplication for the sliding window
│   ├── postprocess/          Package: 10-stage correction pipeline
│   │   ├── __init__.py       (public API)
│   │   ├── hallucinations.py (stage 0)
│   │   ├── voice_commands.py (stages 1-2)
│   │   ├── text_utils.py     (stages 3, 9)
│   │   ├── measurements.py   (stage 4)
│   │   ├── terminology.py    (stage 5)
│   │   ├── medical_dict_match.py (stage 7)
│   │   └── pipeline.py       (orchestrator)
│   └── resources/
│       └── radiology_prompt.txt   Whisper initial prompt
├── core/                     Shared app infrastructure (non-dictation)
│   ├── settings.py
│   └── logging_setup.py
├── ui/
│   ├── app.py                (entry point, re-exports main_window)
│   ├── main_window.py        (MainWindow class, shell logic)
│   ├── views.py              (UI construction: panels, buttons, menus)
│   ├── recording_session.py  (recording control & transcription dispatch)
│   ├── dialogs.py            (dialog windows & user confirmations)
│   ├── web_app.py            (FastAPI interface)
│   └── styles.py             (Qt stylesheets)
├── medical/
│   ├── medical_dict.py
│   ├── critical_findings.py
│   └── macros.py
├── features/
│   ├── accent_corrections.py
│   ├── adaptive_learning.py
│   ├── audit_log.py
│   ├── file_manager.py
│   └── report_manager.py
├── resources/
│   └── medical_terms.txt     Wordlist for medical/medical_dict.py
└── templates/                Plain-text report templates
```

**Data flow:**
```
Mic → Recorder (dictation/audio.py) → WAV (data/temp/)
    → LiveTranscribeWorker (dictation/worker.py) — polls WAV on QThread
    → Transcriber (dictation/transcriber.py) — faster-whisper
    → postprocess pipeline (dictation/postprocess/) — 10-stage corrections
    → MainWindow UI → save/export (features/report_manager.py)
```

---

## Module map

### Dictation pipeline

| File | What it does |
|------|-------------|
| `src/dictation/audio.py` | `Recorder` — sounddevice InputStream → WAV; exposes RMS level for VU meter |
| `src/dictation/transcriber.py` | `Transcriber` — lazy-loads Whisper, transcribes file or numpy array, filters hallucinations |
| `src/dictation/worker.py` | `LiveTranscribeWorker` — sliding-window transcription, commit frontier, adaptive sleep |
| `src/dictation/text_diff.py` | `trim_committed_tail()`, `find_overlap()` — boundary deduplication for the sliding window |
| `src/dictation/resources/radiology_prompt.txt` | Whisper initial prompt (~200 radiology terms) |

### Core (shared infrastructure)

| File | What it does |
|------|-------------|
| `src/core/settings.py` | `Settings` — JSON persistence for user prefs (model, theme, window, etc.) |
| `src/core/logging_setup.py` | `setup_logging()` — centralized logging configuration (called once from entry points) |

### Postprocessing pipeline (10 stages)

| Stage | Module | Purpose |
|-------|--------|---------|
| 0 | `postprocess/hallucinations.py` | Strip Whisper hallucinations ("thank you for watching", repetitions) |
| 1-2 | `postprocess/voice_commands.py` | Voice correction commands, spoken punctuation |
| 3, 9 | `postprocess/text_utils.py` | Normalize spaces, capitalize sentences |
| 4 | `postprocess/measurements.py` | Standardize "5 by 3 mm" → "5 x 3 mm", degrees symbol |
| 5 | `postprocess/terminology.py` | MSK/radiology corrections (tier→tear, anterior→anterior, etc.) |
| 6 | (in pipeline.py) | Accent-specific corrections (south_asian, middle_eastern, etc.) |
| 7 | `postprocess/medical_dict_match.py` | Fuzzy medical dictionary matching (rapidfuzz, 0.92 threshold) |
| 8 | (in pipeline.py) | Learned corrections from adaptive learning |
| – | `postprocess/pipeline.py` | Orchestrator: `postprocess_transcript()`, `postprocess_transcript_with_changes()` |

### UI modules

| File | What it does |
|------|-------------|
| `src/ui/app.py` | Entry point; re-exports `MainWindow` and `main()` from main_window.py |
| `src/ui/main_window.py` | `MainWindow` class — window shell, event dispatch, keyboard shortcuts |
| `src/ui/views.py` | UI construction — `build_ui()`, `build_*_panel()`, `build_menu()` |
| `src/ui/recording_session.py` | Recording control — `on_start_recording()`, `on_stop_recording()`, live transcription callbacks |
| `src/ui/dialogs.py` | Dialog windows — consent, disclaimer, validation, learning stats |
| `src/ui/web_app.py` | FastAPI single-page app; accepts audio blobs, returns transcription JSON |
| `src/ui/styles.py` | `DARK` / `LIGHT` Qt stylesheet strings (Catppuccin Mocha / iOS) |

### Medical & features modules

| File | What it does |
|------|-------------|
| `src/medical/medical_dict.py` | Large radiology term correction dict used by postprocess fuzzy matching |
| `src/medical/macros.py` | Loads `data/macros.json`; exposes `MACROS` dict and `REGION_ORDER` |
| `src/medical/critical_findings.py` | `scan_for_critical_findings()` — NegEx negation parser → `CriticalFinding` list |
| `src/features/accent_corrections.py` | Per-accent regex tables; `apply_accent_corrections(text, accent)`, `suggest_accent()` |
| `src/features/adaptive_learning.py` | `AdaptiveLearning` singleton — learns from edits, writes `data/learned_corrections.json` |
| `src/features/audit_log.py` | Append-only JSON-lines audit log → `data/audit.log` (8-year retention, never auto-delete) |
| `src/features/file_manager.py` | All path functions: `temp_dir()`, `autosave_dir()`, `settings_file()`, `macros_file()` |
| `src/features/report_manager.py` | `autosave_report()`, `save_report_txt()`, `export_to_word()` |

---

## Key design decisions

**Whisper prompt priming.** `_RADIOLOGY_INITIAL_PROMPT` in `transcriber.py` is ~200 radiology
terms fed to Whisper before every transcription. Whisper truncates from the front, so the most
critical terms are at the END of the string.

**Sliding window + commit frontier.** The live worker only transcribes the last `_WINDOW_SEC`
of audio per cycle. Segments that slide off the back are "committed" (frozen text, not
re-processed). This caps latency regardless of recording length.

**Two-layer hallucination filtering.** Segment-level in `transcriber.py` (`_is_hallucination`)
catches per-segment garbage during transcription. Text-level in `postprocess/hallucinations.py`
(`filter_hallucinations`, stage 0) catches patterns spanning segment boundaries. Both layers
are intentional — don't merge them.

**Settings vs file_manager.** `settings.py` owns the schema and access API. `file_manager.py`
owns the file path (`settings_file()`). Path logic lives only in `file_manager.py`.

**Adaptive learning.** `AdaptiveLearning` is a thread-safe singleton with a 30-second write
debounce. User can reset all learned data from the UI.

---

## Import convention

```python
# Always — absolute from project root
from src.dictation.transcriber import Transcriber
from src.features.file_manager import temp_dir

# Never — these break outside src/
from transcriber import Transcriber
from .transcriber import Transcriber
```

Entry points use `python -m src.ui.app` (module mode), which puts project root on `sys.path`.

---

## Data layout

```
data/
├── temp/                    Temp WAVs — cleared on startup
├── autosave/                Auto-saved reports — 30-day retention
├── macros.json              User quick phrases — hot-reloaded
├── audit.log                Append-only audit trail — NEVER delete
└── learned_corrections.json Adaptive learning — user-resetable
```

---

## Coding standards & dependencies

Style rules (naming, type hints, docstrings, imports, logging, errors): see [CODING_STANDARDS.md](CODING_STANDARDS.md).
Runtime deps: see [requirements.txt](requirements.txt).
