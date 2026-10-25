# CLAUDE.md: Radio Dictate

Radio Dictate is an **offline medical dictation workstation** for radiologists.
ASR (automatic speech recognition) runs locally via Parakeet (`onnx-asr`), with
Whisper (`faster-whisper`) behind it; by default **no audio or text leaves the device**. Two front-ends share one
dictation core: a PySide6 desktop GUI and a FastAPI web app.

Read this first. For conventions, see [CODING_STANDARDS.md](CODING_STANDARDS.md).

## Where the detail lives

- What every file is for: [docs/module-map.md](docs/module-map.md). The short
  version is [CORE-FLOW.md](CORE-FLOW.md).
- Six folders keep their guidance in their own `CLAUDE.md`, each loaded only
  when work touches that folder: the dictation pipeline's data-flow and
  live-speed rules (`src/dictation/CLAUDE.md`), the front-ends and their
  diagnostics (`src/ui/CLAUDE.md`), the correction pipeline
  (`src/dictation/postprocess/CLAUDE.md`), cloud fine-tuning
  (`src/cloud/CLAUDE.md`, opt-in), the scan assistant (`src/imaging/CLAUDE.md`,
  opt-in), and the accuracy-measuring harness (`scripts/eval/CLAUDE.md`).
- Any change to transcription or post-processing is judged by that harness's
  numbers, not by reading a sample (`stage_attribution.py` scores each
  post-processing stage on its own; `rescore.py` re-runs the pipeline on stored
  decodes in seconds): read `scripts/eval/CLAUDE.md` and
  [docs/dictation-accuracy.md](docs/dictation-accuracy.md) before proposing an
  accuracy change; several obvious ones are already refuted there.

## Invariants (do not break)

- **Offline by default, one named opt-in.** No network call unless cloud
  training is explicitly enabled *and* consented, or the `asr_engine` setting
  is `deepgram`: that engine sends raw audio to Deepgram's cloud API. The
  default is `local` (2026-09-29): Deepgram's key had been rejected for weeks,
  so every decode was already running on Parakeet, which then measured more
  accurate than the Whisper setup on the live path (4.6 % vs 12.5 % medical-
  term error). Deepgram has no PHI-scrubbing step of its own (audio can't be
  de-identified before it's transcribed), so treat switching it on as a
  BAA/compliance question. It falls back to the local chain on any network or
  auth failure (`engines/fallback_engine.py`). Everything else in
  `dictation/` stays offline; cloud training itself never imports cloud.
- **PHI is scrubbed before anything leaves the device** via `DeIdentifier` +
  `validate_clean()` (raises `PrivacyError`); a record that still contains a
  known identifier is dropped, not uploaded.
- **Untrusted artifacts are validated**: a downloaded model archive is
  extracted only after every member is confirmed to resolve inside the
  destination (`cloud/framework/sync.py::_extract_model`).
- **Secrets live in the OS keychain only** (`keyring`), never in
  `dictation_settings.json` or logs. The Lightning project id (not a secret)
  is in settings; the Groq key follows the same rule
  (`llm_cleanup.get_groq_key`).
- **Scan suggestions abstain by default and must be localisable**: a finding
  reaches the radiologist only if its label was trained AND its probability ≥
  the calibrated threshold AND Grad-CAM produced a region. Always an assistive
  suggestion with a non-diagnostic disclaimer: never a diagnosis.
- **AI cleanup is off by default, scrubbed, reviewed, and lossless on
  failure**: Groq cleanup runs only when enabled + consented, de-identifies
  before sending (with the patient's own name and ID), edits only language
  (never clinical content), returns the input unchanged on any error, and is
  applied only after the radiologist accepts it (`ui/dialogs.offer_ai_cleanup`,
  the one path, from the menu or the 'hard' level). The pipeline never calls it.
- **No rule swaps one diagnosis for another.** A rule may fix a mishearing
  ("menescus" → "meniscus") or a spelling convention, never a word the
  radiologist chose ("tendinitis" stays "tendinitis").
- **Settings file is `dictation_settings.json` at the project root.**
- **File I/O goes through `core/paths.py`** path helpers
  (`autosave_dir`, `staging_db_path`, `model_registry_path`, `fine_tuned_dir`,
  `imaging_thresholds_path`, `imaging_overlay_dir`, …).
- **Every rebuildable cache lives under `data/cache/`** (`cache_dir()`:
  SymSpell index, Whisper model downloads via `whisper_cache_dir()`, imaging
  embeddings via `imaging_embeddings_dir()`). Deleting the folder is always
  safe: caches rebuild or re-download on next use. Never write derived,
  regenerable data anywhere else.

## Run & test

```bash
python -m src.ui            # desktop GUI
python -m src.ui.web_app    # web app on 127.0.0.1:8005
ruff check src tests scripts
python -m pytest -q          # safety tables + streaming rules; CI runs both
npx pyright src              # type check (optional-dep import warnings expected)

# Accuracy + speed measurement: see scripts/eval/CLAUDE.md

# Optional extras (lazy-imported; core app runs without them):
pip install -e ".[imaging]"     # Scan Assistant (local)
pip install -e ".[parakeet]"    # Parakeet ASR engine (local)
pip install groq                                            # AI Cleanup
```