# Radio Dictate

Offline medical dictation workstation for radiologists. Speech-to-text powered by
local Whisper (`faster-whisper`) — no audio or text leaves the machine.

Two interfaces:
- **Desktop** — PySide6 GUI with templates, macros, and live transcription
- **Web** — FastAPI single-page app for browser-based dictation

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate                    # Windows
# source .venv/bin/activate               # macOS / Linux
pip install -r requirements.txt

python -m src.ui.app                      # Desktop GUI
python -m src.ui.web_app                  # Web interface
python scripts/verify_setup.py            # Optional dependency check
```

> Always run from the **project root**. `src.*` imports break if you `cd` into `src/`.

The Whisper model downloads automatically on first transcription and is cached
in your user directory.

## Requirements

- Python 3.10 or 3.11
- A microphone
- NVIDIA GPU (optional — required only for `medium` / `large-v3` models)

## Features

- **Live transcription** — text streams while you dictate
- **Radiology-tuned Whisper prompt** — ~200 domain terms prime accuracy
- **10-stage post-processing** — hallucinations → voice commands → punctuation → measurements → terminology → accent-specific → fuzzy match → learned corrections → capitalization
- **Medical dictionary** — large term list with rapidfuzz fuzzy matching (~92% threshold)
- **Templates** — chest, neuro, abdominal, MSK, ultrasound (`src/templates/*.txt`)
- **Macros** — quick phrases by region, hot-reloaded from `data/macros.json`
- **Word export** — formatted `.docx` with patient info table
- **Auto-save** — backups to `data/autosave/` (30-day retention)
- **Adaptive learning** — passively learns from your edits
- **Critical findings detection** — NegEx negation parser flags urgent results
- **Audit log** — append-only trail (`data/audit.log`, 8-year retention)
- **Dark / light themes** — Catppuccin Mocha and iOS-inspired

## Keyboard shortcuts

| Shortcut       | Action                        |
|----------------|-------------------------------|
| F5             | Start recording               |
| F6             | Stop recording                |
| Ctrl+S         | Save report as text           |
| Ctrl+Shift+W   | Export to Word (.docx)        |
| Ctrl+T         | Load template                 |
| Ctrl+R         | Reload macros                 |
| Ctrl+D         | Toggle dark / light theme     |
| Ctrl+P         | Toggle patient panel          |
| Ctrl+M         | Toggle macros panel           |
| Ctrl+] / Ctrl+[| Increase / decrease font size |

## Voice commands while dictating

- Punctuation: `"full stop"`, `"comma"`, `"new line"`, `"new paragraph"`, `"colon"`, `"hyphen"`
- Inline correction: `"correct word X"` replaces the previous word with X

## Project layout

```
src/
├── core/
│   ├── audio.py · transcriber.py · settings.py · text_diff.py · logging_setup.py
│   └── postprocess/          10-stage correction pipeline package
│       ├── hallucinations.py · voice_commands.py · text_utils.py
│       ├── measurements.py · terminology.py · medical_dict_match.py
│       └── pipeline.py       (orchestrator)
├── ui/
│   ├── app.py (entry point) · main_window.py · views.py
│   ├── recording_session.py · dialogs.py
│   └── web_app.py (FastAPI) · styles.py
├── medical/
│   └── medical_dict.py · critical_findings.py · macros.py
├── workers/
│   └── transcribe_worker.py  (live transcription QThread)
├── features/
│   ├── accent_corrections.py · adaptive_learning.py · audit_log.py
│   └── file_manager.py · report_manager.py
├── templates/                plain-text report templates
└── resources/                medical_terms.txt wordlist

data/         temp WAVs · autosave reports · macros.json · audit.log
config/       dictation_settings.json
scripts/      verify_setup.py · run_web.bat · update_script.py
tests/        pytest suite (103 tests)
.claude/      Claude Code skills files
```

For architecture details, design decisions, and module API, see
[CLAUDE.md](CLAUDE.md). For style rules, see [CODING_STANDARDS.md](CODING_STANDARDS.md).

## Customisation

- **Templates** — drop a new `.txt` file in `src/templates/`
- **Macros** — edit `data/macros.json`, reload with Ctrl+R
- **Post-processing**
  - **Terminology corrections** — edit `src/core/postprocess/terminology.py`
  - **Accent-specific corrections** — edit `src/features/accent_corrections.py`
  - **Hallucinations** — edit `src/core/postprocess/hallucinations.py`
  - **Measurements** — edit `src/core/postprocess/measurements.py`
- **Fuzzy matching cutoff** — tune `cutoff=0.92` in `src/core/postprocess/medical_dict_match.py` (lower = more aggressive)
- **Medical dictionary** — add terms to `src/resources/medical_terms.txt`

## Troubleshooting

- **Microphone errors** — check input device in OS Sound Settings; close other apps using the mic
- **Import errors** — activate venv and reinstall: `pip install -r requirements.txt`
- **Slow transcription** — switch to a smaller model (`tiny` / `base`) or check logs for `compute_type`
- **Over-correction** — raise the fuzzy cutoff in `postprocess.py` or add the term to `_PROTECTED_TERMS`
- **Missing Word export** — `pip install python-docx`

## Tests

```bash
python -m pytest tests/
```

## Notes

- All processing is local. No audio or text is transmitted off-device.
- Whisper models cache in `~/.cache/huggingface/`.
- The audit log is append-only and never auto-deleted (retain for 8 years).
