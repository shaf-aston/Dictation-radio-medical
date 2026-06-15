# CLAUDE.md — Architecture & Module Map

Radio Dictate is an **offline medical dictation workstation** for radiologists.
Speech-to-text runs locally via Whisper (`faster-whisper` / CTranslate2); by
default **no audio or text leaves the device**. Two front-ends share one
dictation core: a PySide6 desktop GUI and a FastAPI web app.

Read this first. For conventions, see [CODING_STANDARDS.md](CODING_STANDARDS.md).
For project-specific automation, see **Tooling** below — check it before doing
manual multi-file work that an existing agent/skill/workflow already covers.

## Tooling — `.claude/` (check this before manual multi-step work)

```
.claude/
├── agents/      sub-agents — delegate research/review here to keep the
│                 main context clean (separate context window, returns a summary)
├── skills/      on-demand procedures — loaded only when their description
│                 matches the task; no cost until triggered
└── workflows/   multi-step pipelines composing the above
```

If a task matches an existing agent/skill/workflow's stated purpose, use it
instead of re-deriving the steps inline. If something here *should* trigger
but doesn't, the fix is almost always the `description` field in that item's
own frontmatter (too vague → never fires; too narrow → only fires for an exact
phrasing) — not this file. New session required after adding/editing a skill
or agent for it to be picked up.

## Module map

```
src/
├── core/        settings.py (JSON at project root) · logging_setup.py
├── dictation/   the offline pipeline — has NO cloud dependency
│   ├── audio.py          microphone capture
│   ├── worker.py         live transcription QThread (sliding window)
│   ├── transcriber.py    faster-whisper / CTranslate2 wrapper
│   ├── text_diff.py      incremental diff for streaming UI updates
│   ├── postprocess/      10-stage correction pipeline (pipeline.py orchestrates)
│   └── resources/        radiology_prompt.txt (Whisper priming prompt, ~200 terms)
├── ui/          main_window.py · views.py · recording_session.py · dialogs.py
│   ├── web_app.py        FastAPI single-page app (binds 127.0.0.1:8005)
│   ├── styles.py · styles/*.qss · frontends/   desktop + web assets
│   └── __main__.py       enables `python -m src.ui`
├── medical/     medical_dict.py · critical_findings.py (NegEx) · macros.py
├── features/    accent_corrections.py · adaptive_learning.py · audit_log.py
│   └── file_manager.py · report_manager.py · report_analyzer.py
├── imaging/     OPTIONAL local chest X-ray assistant (offline inference)
│   ├── classifier.py     TorchXRayVision DenseNet121 wrapper (lazy torch)
│   ├── abstention.py     calibrated rejection gate — the imaging safety gate
│   ├── localization.py   hand-rolled Grad-CAM → region + overlay PNG
│   ├── analyzer.py       orchestrator: classify → abstain → localize → result
│   ├── schemas.py        ImagingFinding / ImagingResult / DISCLAIMER
│   └── resources/        thresholds.json (default per-pathology cutoffs)
├── cloud/       OPTIONAL Lightning AI fine-tuning (consent-gated)
│   ├── framework/        task-agnostic core
│   │   ├── client.py     REST wrapper; API key in OS keychain (keyring)
│   │   ├── registry.py   task-keyed source of truth for active models
│   │   ├── sync.py       upload → train → download → register façade
│   │   └── job_monitor.py QThread that polls jobs and signals readiness
│   ├── tasks/            one plug-in per model type (per-task differences only)
│   │   ├── base.py       TrainingTask Protocol + JobSpec
│   │   └── whisper_voice.py · text_corrector.py · scan_finetune.py
│   ├── privacy.py        PHI de-identification — the upload safety gate
│   └── exceptions.py     CloudError hierarchy (+ ImagingError, GroqError)
├── training/    collector.py · schemas.py · staging_db.py (SQLite)
├── templates/   plain-text report templates (RSNA / MSK / generic)
└── resources/   medical_terms.txt (broad generic wordlist — membership net) ·
                  radiology_lexicon.txt (curated radiology terms — the clean
                  spelling-correction snap targets)
```

Other optional, off-by-default add-ons:
- `dictation/postprocess/llm_cleanup.py` — on-demand Groq report polish (NOT in
  the per-chunk pipeline); de-identifies first, key in keychain, consent-gated.
- `scripts/lightning/` — training entrypoints run ON Lightning AI: `train_whisper`
  + `convert_to_ct2` (voice), `train_text_corrector`, `train_scan_classifier`.
  Their requirements are separate optional extras (`requirements_*.txt`).

## Dictation data-flow (always local)

```
microphone → audio.py → worker.py (QThread, sliding window)
           → transcriber.py (Whisper) → postprocess/ (10 stages)
           → UI (views.py / web_app.py) → report_manager.py (.docx / .txt export)
```

The pipeline (`dictation/postprocess/pipeline.py`) runs, in order: hallucination
removal → voice commands → punctuation → measurements → terminology →
accent-specific → fuzzy medical-dictionary match → learned corrections →
capitalization. Each stage owns one file; the pipeline only sequences them.

The fuzzy stage (`medical_dict_match.py`) is where mis-transcribed medical terms
get fixed, and it leans on **two** wordlists with distinct jobs (`medical_dict.py`):
the broad generic list answers *"is this already a real word? leave it alone"*
(membership), while the **curated `radiology_lexicon.txt`** is the only thing a
typo is *snapped to* (correction targets). Keeping snap targets radiology-only is
what stops a misspelling from being pulled toward the generic list's chemistry /
drug / obscure-procedure junk. To improve correction of a term, add it to the
lexicon (the spelling authority) or add a precise rule to `corrections.yaml`.

## Cloud fine-tuning path (opt-in, off by default)

Inert unless **both** `cloud_enabled` and `cloud_training_consent` are true in
settings. With default settings nothing is retained, staged, or uploaded.

```
dictation corrections → training/collector.py (consent-gated capture)
   → cloud/privacy.py  de-identify TEXT + AUDIO, validate_clean()
   → training/staging_db.py  (SQLite: data/training/staging.db)
   → cloud/tasks/<task>.build_archive()  tar.gz batch (manifest + de-id clips)
   → cloud/framework/client.py  upload + submit Lightning AI job (key from keychain)
   → cloud/framework/job_monitor.py polls → framework/sync.py downloads + extracts
   → cloud/framework/registry.py registers the version (data/models/registry.json)
   → user activates it; the consuming model loads that artifact directory
```

`collector.py` is the **only** bridge from dictation into the cloud subsystem,
and the dependency is one-directional: dictation/features call into the
collector but never import `src.cloud.*`.

**Multi-task framework.** The framework is task-agnostic; a `TrainingTask`
(`cloud/tasks/`) supplies only what differs per model — how to bundle its batch
and how to describe its Lightning job (`JobSpec`). Three tasks exist: voice
(Whisper LoRA → CT2), text-correction (small seq2seq), and scan-classifier
(DenseNet head fine-tune). Each is keyed by `task_type`; the registry holds one
active model **per task**. Every training script enforces a *do-not-regress*
gate on a held-out split (WER for voice, exact-match for text, mean-AUC for
scans) and rejects a run that doesn't beat its validated baseline.

## Scan assistant path (opt-in, local inference)

`src/imaging/` analyses a chest X-ray locally and surfaces findings **only when
confident and localisable**. Pipeline: `classifier` (TorchXRayVision DenseNet121)
→ `abstention` (drop untrained heads + anything below the calibrated
per-pathology threshold) → `localization` (Grad-CAM region per kept finding) →
`analyzer` (withhold any finding it can't point to; render an overlay; attach
the non-diagnostic disclaimer). Needs the optional imaging extra
(`scripts/lightning/requirements_imaging.txt`); without it the feature degrades
gracefully and the rest of the app is unaffected.

**Specialty focus: chest trauma.** Rib/clavicle fractures correlate clinically
with pneumothorax and haemothorax, and plain films are documented to miss a
large share of rib fractures — the highest-leverage gap for this assistant to
close. The local fine-tune (`scan_classifier` task, `ScanClassifierTask` in
`cloud/tasks/scan_finetune.py`) oversamples `Fracture`, `Pneumothorax`, and
`Effusion` positives (`_TRAUMA_FOCUS_LABELS`) when building a batch, so a site's
fine-tune sharpens on trauma cases rather than diluting evenly across all 18
baseline labels. This changes *what the model trains on*, not the safety gate —
per-label abstention thresholds remain the calibration lever for a site's own
validated data (`data/imaging/thresholds.json`).

## Invariants (do not break)

- **Offline by default.** No network call unless cloud training is explicitly
  enabled *and* consented. Dictation never imports cloud.
- **PHI is scrubbed before anything leaves the device** via `DeIdentifier` +
  `validate_clean()` (raises `PrivacyError`); a record that still contains a
  known identifier is dropped, not uploaded.
- **Untrusted artifacts are validated** — a downloaded model archive is
  extracted only after every member is confirmed to resolve inside the
  destination (`cloud/framework/sync.py::_extract_model`).
- **Secrets live in the OS keychain only** (`keyring`), never in
  `dictation_settings.json` or logs. The Lightning project id (not a secret)
  is in settings; the Groq key follows the same rule
  (`llm_cleanup.get_groq_key`).
- **Scan suggestions abstain by default and must be localisable** — a finding
  reaches the radiologist only if its label was trained AND its probability ≥
  the calibrated threshold AND Grad-CAM produced a region. Always an assistive
  suggestion with a non-diagnostic disclaimer — never a diagnosis.
- **AI cleanup is off by default, scrubbed, and lossless on failure** — Groq
  cleanup runs only when enabled + consented, de-identifies before sending,
  edits only language (never clinical content), and returns the input
  unchanged on any error.
- **Settings file is `dictation_settings.json` at the project root.**
- **File I/O goes through `features/file_manager.py`** path helpers
  (`autosave_dir`, `staging_db_path`, `model_registry_path`, `fine_tuned_dir`,
  `imaging_thresholds_path`, `imaging_overlay_dir`, …).

## Run & test

```bash
python -m src.ui            # desktop GUI
python -m src.ui.web_app    # web app on 127.0.0.1:8005
python scripts/verify_setup.py
python -m pytest tests/ -q
ruff check src tests
npx pyright src              # type check (optional-dep import warnings expected)

# Optional extras (lazy-imported; core app runs without them):
pip install -r scripts/lightning/requirements_imaging.txt   # Scan Assistant (local)
pip install groq                                            # AI Cleanup
```