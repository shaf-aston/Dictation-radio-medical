# Radio Dictate

Offline medical dictation workstation for radiologists. Speech-to-text via local Whisper (`faster-whisper`) — no audio or text leaves the machine.

Two interfaces:
- **Desktop** — PySide6 GUI with templates, macros, and live transcription
- **Web** — FastAPI single-page app at `http://127.0.0.1:8005`

## Quick start

```bash
python -m src.ui                   # desktop GUI
python -m src.ui.web_app           # web app
python scripts/verify_setup.py    # verify installation
python -m pytest tests/ -q        # run tests
ruff check src tests               # lint
```

> Always run from the **project root**. `src.*` imports break if you `cd` into `src/`.

## Features

- **Live transcription** — text streams while you dictate
- **Radiology-tuned Whisper prompt** — ~200 domain terms prime accuracy
- **10-stage post-processing** — hallucination removal → voice commands → punctuation → measurements → terminology → accent-specific → fuzzy match → learned corrections → capitalization
- **Medical dictionary** — rapidfuzz fuzzy matching (~92% threshold)
- **Templates** — chest, neuro, abdominal, MSK, ultrasound
- **Macros** — quick phrases by region, hot-reloaded from `data/macros.json`
- **Word export** — formatted `.docx` with patient info table
- **Auto-save** — backups to `data/autosave/` (30-day retention)
- **Adaptive learning** — passively learns from your edits
- **Critical findings detection** — NegEx negation parser flags urgent results
- **Audit log** — append-only trail (`data/audit.log`, 8-year retention)
- **Dark / light themes** — Catppuccin Mocha and iOS-inspired

## Keyboard shortcuts

| Shortcut         | Action                         |
|------------------|--------------------------------|
| F5               | Start recording                |
| F6               | Stop recording                 |
| Ctrl+S           | Save report as text            |
| Ctrl+Shift+W     | Export to Word (.docx)         |
| Ctrl+T           | Load template                  |
| Ctrl+R           | Reload macros                  |
| Ctrl+D           | Toggle dark / light theme      |
| Ctrl+P           | Toggle patient panel           |
| Ctrl+M           | Toggle macros panel            |
| Ctrl+] / Ctrl+[  | Increase / decrease font size  |

## Voice commands

- Punctuation: `"full stop"`, `"comma"`, `"new line"`, `"new paragraph"`, `"colon"`, `"hyphen"`
- Inline correction: `"correct word X"` replaces the previous word with X

## Project layout

```
src/
├── core/                settings.py · logging_setup.py
├── dictation/           audio.py · transcriber.py · text_diff.py · worker.py
│   ├── postprocess/     pipeline.py · hallucinations.py · voice_commands.py · text_utils.py
│   │                    measurements.py · terminology.py · medical_dict_match.py · analysis.py
│   └── resources/       radiology_prompt.txt
├── ui/                  main_window.py · views.py · recording_session.py · dialogs.py
│   ├── web_app.py       FastAPI single-page app
│   └── styles.py · styles/*.qss · frontends/
├── medical/             medical_dict.py · critical_findings.py · macros.py
├── features/            accent_corrections.py · adaptive_learning.py · audit_log.py
│   └── file_manager.py · report_manager.py · report_analyzer.py
├── cloud/               Lightning AI fine-tuning (opt-in): client · privacy · uploader
│   └── sync_manager.py · job_monitor.py · model_registry.py · exceptions.py
├── training/            collector.py · schemas.py · staging_db.py
├── templates/           plain-text report templates (RSNA / MSK / generic)
└── resources/           medical_terms.txt

data/                    temp WAVs · autosave/ · macros.json · audit.log · training/ · medical_reference/
dictation_settings.json  app settings
scripts/                 verify_setup.py · lightning/ (cloud training) · download_medical_references.py
tests/                   pytest suite
```

See [CLAUDE.md](CLAUDE.md) for architecture details, [CODING_STANDARDS.md](CODING_STANDARDS.md) for style rules, and [MEDICAL_REFERENCE_INTEGRATION.md](MEDICAL_REFERENCE_INTEGRATION.md) for training corpus setup.

### Medical Reference Library

Radio Dictate includes open-access radiology textbooks in `data/medical_reference/` for terminology extraction, clinical reasoning, and synthetic training data generation:

- **A to Z of Chest Radiology** (Misra et al.) — chest trauma, pneumothorax, rib fractures
- **A to Z of Emergency Radiology** (Holmes & Misra) — acute pathology patterns
- **Basic Radiology** (Chen et al.) — fundamental imaging principles
- **Principles of Radiographic Imaging** (Carlton et al.) — radiographic physics

See [data/ORGANIZATION.md](data/ORGANIZATION.md) for folder structure and [data/medical_reference/metadata.json](data/medical_reference/metadata.json) for corpus inventory.

## Customisation

- **Templates** — drop a `.txt` file in `src/templates/`
- **Macros** — edit `data/macros.json`
- **Terminology / hallucinations / measurements / accent corrections** — edit the corresponding file in `src/dictation/postprocess/` or `src/features/accent_corrections.py`
- **Fuzzy matching cutoff** — tune `cutoff=0.92` in `src/dictation/postprocess/medical_dict_match.py`; add protected terms to `_PROTECTED_TERMS`
- **Medical dictionary** — add terms to `src/resources/medical_terms.txt`

## Troubleshooting

- **Microphone errors** — check input device in OS Sound Settings; close other apps using the mic
- **Import errors** — activate venv: `pip install -r requirements.txt`
- **Slow transcription** — switch to a smaller Whisper model (`tiny` / `base`)
- **Over-correction** — raise the fuzzy cutoff or add the term to `_PROTECTED_TERMS` (see Customisation)
- **Missing Word export** — `pip install python-docx`

## Notes

- The audit log is append-only; retain for 8 years per clinical record requirements.
- Whisper models cache in `~/.cache/huggingface/` on first use.
