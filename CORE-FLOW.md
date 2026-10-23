# Radio Dictate

Lets a radiologist dictate a report by voice: speech is transcribed and auto-corrected offline, entirely on the radiologist's own machine.

## Run it

```bash
pip install -e .                     # deps from pyproject.toml into the venv
python -m src.ui                     # desktop GUI, verified: builds and paints in both themes
python -m src.ui.web_app             # web app on http://127.0.0.1:8005, verified: starts, GET / returns 200
```

## Core flow

**A. Dictation (live, both front-ends)**

1. **Record (F5, or F2 to start and stop) captures the microphone**: the desktop writes a growing WAV (`src/dictation/audio.py`), the browser streams 16 kHz PCM over `/ws/dictate`. Both feed the same loop, `src/dictation/stream/live_session.py`.
2. **The still-open tail is VAD-cut into chunks and each is decoded exactly once** by Parakeet, with Whisper behind it (engine behind `src/dictation/asr/`). Exists so a long dictation doesn't cost multiples of its own length in redundant re-decoding.
3. **The new (uncommitted) text runs through the correction chain**: `src/dictation/postprocess/pipeline.py`. Raw recogniser output contains hallucinations, spoken punctuation and commands ("scratch that", "insert <phrase>"), and medical mis-hearings that must become a clean report.
4. **The text goes in at the caret, or over a selected template field**: `src/ui/recording_session.py` (desktop), `src/ui/frontends/app.js` (web). Text after the caret is never rewritten. Saying one option of a selected pick-list (`[Right / Left]`) answers it (`src/features/template_fields.py`). Tab / Shift+Tab move between fields.
5. **Stop hands the text back at once; a confidence-targeted polish follows** and is dropped if the report was edited meanwhile.
6. **Every way the report leaves is gated**: `src/features/report_release.py`. Copy, Save TXT and Export Word ask one question about unfilled fields and consistency (side, sex, impression: `src/features/report_checks.py`), which can send the radiologist back, then about critical findings, which records who was told, how and when, and can add that sentence to the report. The report is never withheld; only the audit entry differs.
7. **The report is saved or exported** to `.txt`/`.docx`: `src/features/report_manager.py`. Every exit is audited (`src/features/audit_log.py`).

## Glossary

- **Chunk-once ledger**: the point behind which live text is frozen and never re-decoded; each chunk closes at a VAD silence boundary (`src/dictation/stream/ledger.py`, `ChunkPolicy.min_sec`/`soft_max_sec`/`force_cut_sec`).
- **Postprocess pipeline**: the ordered 10-stage cleanup chain (`src/dictation/postprocess/pipeline.py`) that turns raw Whisper text into a formatted report.
- **Open tail**: the still-growing slice of audio re-decoded each cycle for a stable live preview only (never committed), instead of the whole growing recording.
- **Medical dictionary (fuzzy match)**: corrects mis-heard medical terms by snapping close matches to a curated term list (`src/medical/medical_dict.py`).
- **Radiology lexicon vs. medical terms list**: two wordlists: one just recognizes real words (leave alone), the other is what a typo gets corrected *to* (`src/resources/`).
- **Macros**: short phrases that expand into boilerplate report text, hot-reloaded from `data/macros.json`.
- **Critical findings**: a rule-based (NegEx) scan that flags urgent results before sign-off (`src/medical/critical_findings.py`). The gate that runs it on every exit is `src/features/report_release.py`; both front-ends call that one, so they cannot drift.
- **De-identification (PHI scrub)**: stripping patient-identifying info before anything leaves the device; only exercised if cloud training is opted in (`src/medical/deid.py`).
- **Templates**: starter report text per exam type (chest, MSK, etc.), in `src/templates/`.
- **Template field**: any bracketed span on one line of a template, `[Right / Left]` or `[X]`; declared once in `src/features/template_fields.py` for the release gate, Tab navigation and pick-list answers.
- **Design tokens**: the 11 named colours in `src/ui/tokens.json`. Both front-ends render their stylesheets from this one file (`src/ui/theme.py`), so the desktop app and the web app always look like the same product. Red means recording or clinical severity; cyan is the machine's voice; nothing else is coloured.
- **Abstention**: the optional X-ray assistant's rule to stay silent unless confident and able to localize a finding (`src/imaging/abstention.py`).

## Where things live

- `src/`: all application code: `core` (settings/logging), `dictation` (offline pipeline), `ui` (desktop + web front-ends), `medical`, `features`, `imaging` (optional X-ray assistant), `cloud` (optional fine-tuning), `training`, `templates`.
- `data/`: runtime data: autosave backups, cache, `macros.json`, `audit.log`, training corpus.
- `scripts/`: one-off utilities: correction mining, accuracy evaluation (`eval/`), cloud training entrypoints.
- `CLAUDE.md`, full per-module map and the live-speed design.
- `dictation_settings.json`: the one settings file, read at project root.
