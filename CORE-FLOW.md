# Radio Dictate

Lets a radiologist dictate a report by voice — speech is transcribed and auto-corrected offline, entirely on the radiologist's own machine.

## Run it

```bash
pip install -r requirements.txt      # (unverified) install deps into a venv
python -m src.ui                     # desktop GUI (import verified only — not launched)
python -m src.ui.web_app             # web app on http://127.0.0.1:8005 — verified: starts, GET / returns 200
```

## Core flow

**A. Desktop dictation (live)**

1. **Record (F5) captures the microphone** to a growing WAV file — `src/dictation/audio.py`. Exists so audio capture never blocks transcription.
2. **The still-open tail is VAD-cut into chunks and each is decoded exactly once** by Whisper via `src/dictation/worker.py` (VAD/segmenter/ledger in `src/dictation/stream/`, engine behind `src/dictation/asr/`). Exists so a long dictation doesn't cost multiples of its own length in redundant re-decoding — only the still-open tail is ever re-decoded, for a stable live preview.
3. **The new (uncommitted) text runs through a 10-stage cleanup chain** — `src/dictation/postprocess/pipeline.py`. Exists because raw Whisper output contains hallucinations, spoken punctuation/commands, and medical mis-hearings that must become a clean report.
4. **The editor updates live** — `src/ui/recording_session.py` / `src/ui/views.py`. Exists so the radiologist sees (and can correct) text as they speak.
5. **Stop (F6) triggers a confidence-targeted polish**, not a full re-transcription — `src/dictation/worker.py::_run_confidence_targeted_polish`. Only committed chunks whose mean word confidence fell below the ceiling (plus whatever was still open) get one higher-beam re-decode; already-confident chunks are never touched again.
6. **The finished text is scanned for urgent findings and logged** — `src/medical/critical_findings.py`, `src/features/audit_log.py`. Exists so nothing urgent is missed and every report leaves an audit trail.
7. **The report is saved or exported** to `.txt`/`.docx` — `src/features/report_manager.py`. Exists to hand the finished report into the radiologist's normal workflow.

**B. Web dictation (one-shot)**

1. The browser records the entire session client-side and POSTs the audio once, on stop — `src/ui/web_app.py`. Exists so the browser never needs the live sliding-window machinery.
2. The server transcribes the full audio once and runs it through the **same** 10-stage pipeline as desktop. Exists so terminology/corrections behave identically regardless of which front-end is used.

## Glossary

- **Chunk-once ledger** — the point behind which live text is frozen and never re-decoded; each chunk closes at a VAD silence boundary (`src/dictation/stream/ledger.py`, `ChunkPolicy.min_sec`/`soft_max_sec`/`force_cut_sec`).
- **Postprocess pipeline** — the ordered 10-stage cleanup chain (`src/dictation/postprocess/pipeline.py`) that turns raw Whisper text into a formatted report.
- **Open tail** — the still-growing slice of audio re-decoded each cycle for a stable live preview only (never committed), instead of the whole growing recording.
- **Medical dictionary (fuzzy match)** — corrects mis-heard medical terms by snapping close matches to a curated term list (`src/medical/medical_dict.py`).
- **Radiology lexicon vs. medical terms list** — two wordlists: one just recognizes real words (leave alone), the other is what a typo gets corrected *to* (`src/resources/`).
- **Macros** — short phrases that expand into boilerplate report text, hot-reloaded from `data/macros.json`.
- **Critical findings** — a rule-based (NegEx) scan that flags urgent results before sign-off (`src/medical/critical_findings.py`).
- **De-identification (PHI scrub)** — stripping patient-identifying info before anything leaves the device; only exercised if cloud training is opted in (`src/medical/deid.py`).
- **Templates** — starter report text per exam type (chest, MSK, etc.), in `src/templates/`.
- **Design tokens** — the 11 named colours in `src/ui/tokens.json`. Both front-ends render their stylesheets from this one file (`src/ui/theme.py`), so the desktop app and the web app always look like the same product. Red means recording or clinical severity; cyan is the machine's voice; nothing else is coloured.
- **Abstention** — the optional X-ray assistant's rule to stay silent unless confident and able to localize a finding (`src/imaging/abstention.py`).

## Where things live

- `src/` — all application code: `core` (settings/logging), `dictation` (offline pipeline), `ui` (desktop + web front-ends), `medical`, `features`, `imaging` (optional X-ray assistant), `cloud` (optional fine-tuning), `training`, `templates`.
- `data/` — runtime data: autosave backups, cache, `macros.json`, `audit.log`, training corpus.
- `scripts/` — one-off utilities: setup verification, correction mining, cloud training entrypoints.
- `docs/ARCHITECTURE.md` — deeper flow + performance notes; `CLAUDE.md` — full per-module map.
- `dictation_settings.json` — the one settings file, read at project root.
- `tests/` — existing pytest suite (not touched by this pass).
