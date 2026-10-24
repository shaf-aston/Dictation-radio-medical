# CLAUDE.md: Architecture & Module Map

Radio Dictate is an **offline medical dictation workstation** for radiologists.
ASR (automatic speech recognition) runs locally via Parakeet TDT v2 (sherpa-onnx,
biased toward the radiology lexicon with hotwords; `onnx-asr` for a fine-tune), with
Whisper (`faster-whisper`) behind it; by default **no audio or text leaves the device**. Two front-ends share one
dictation core: a PySide6 desktop GUI and a FastAPI web app.

Read this first. For conventions, see [CODING_STANDARDS.md](CODING_STANDARDS.md).

## Module map

```
src/
├── core/        settings.py (JSON at project root) · paths.py (every file path
│                  the app uses: data/, cache/, templates; one place, so the
│                  data layout and a future per-user data dir have one seam) ·
│                  logging_setup.py ·
│                  json_store.py (shared atomic JSON + JSONL read/write) ·
│                  keychain.py (OS-keychain secret helpers, wraps keyring) ·
│                  patient_schema.py (the patient-info fields, declared once) ·
│                  perf.py (stage timings: rolling count/mean/p95/max, local only) ·
│                  event_log.py (the diary: a bounded ring of what just happened,
│                    with a duration on anything that took time. perf answers
│                    "what does this stage usually cost"; this answers "what
│                    happened just now, in what order": the developer console
│                    needs both. Mirrors ordinary log lines in too, so the
│                    warning and the timing that explains it sit together)
├── dictation/   the offline pipeline, has NO cloud dependency
│   ├── audio.py          microphone capture
│   ├── worker.py         desktop QThread: thin adapter feeding LiveSession
│   │                       the new samples of the growing WAV (never re-reads)
│   ├── asr/               the AsrEngine swap-seam: port.py
│   │                       (Protocol, plus StreamingAsrEngine/AsrStream and
│   │                       engine_identity) · types.py (Word/AsrSegment/
│   │                       AsrResult/TranscribeContext, confidence is part of
│   │                       the contract; EngineCaps: cost, streaming,
│   │                       network) · factory.py (create_engine, the only
│   │                       name→engine mapping; the asr_engine setting picks
│   │                       one, default "local": parakeet (if installed) →
│   │                       faster-whisper; "deepgram" puts the cloud first.
│   │                       Each result names the provider that made it)
│   │                       · engines/deepgram_engine.py (cloud, nova-2-medical,
│   │                       key in OS keychain, the only network-dependent
│   │                       engine here — see the invariants note below) ·
│   │                       engines/deepgram_stream.py (its live WebSocket:
│   │                       interim + settled text as audio arrives) ·
│   │                       engines/fallback_engine.py (ChainEngine: tries each
│   │                       provider in order, degrades past any that raises;
│   │                       a dead key is skipped until restart, a flaky one
│   │                       by a 30s circuit breaker) ·
│   │                       engines/faster_whisper_engine.py (the CTranslate2
│   │                       wrapper and its port adapter, one file) ·
│   │                       engines/parakeet_hotword_engine.py (the default:
│   │                       sherpa-onnx beam search boosted by the 870-term
│   │                       radiology lexicon; measured 35% fewer term errors
│   │                       than greedy, same speed) ·
│   │                       engines/parakeet_engine.py (onnx-asr greedy; loads a
│   │                       fine-tuned export from scripts/finetune/; no prompt,
│   │                       one model shared process-wide) · prompt.py
│   │                       (the radiology priming vocabulary, same for every
│   │                       engine) · models.py (known model names + fallback)
│   ├── stream/             chunk-once streaming, vad.py (Silero VAD, bundled
│   │                       with faster-whisper, no new dep) · segmenter.py
│   │                       (pure VAD-marks→chunk-cuts policy) · policy.py
│   │                       (chunk sizes from the engine's cost; LiveKnobs, the
│   │                        eight dictation settings both front-ends hand a
│   │                        LiveSession, read in one place) · live_session.py
│   │                       (the Qt-free live loop: a push-fed audio buffer
│   │                        instead of the desktop's growing WAV, so the web
│   │                        app streams over a WebSocket using these same
│   │                        chunk rules; plus socket mode and background
│   │                        polish) · rules.py (build_context_prompt /
│   │                        mean_confidence / should_skip_preview) · ledger.py
│   │                       (freezes each closed chunk's decode permanently,
│   │                       the "decode once" guarantee) · tail.py
│   │                       (LocalAgreement-2 stable preview of the open tail)
│   │                       · polish.py (the polish after Stop: which chunks the
│   │                        accurate model redoes, one rule for both front-ends)
│   ├── postprocess/      10-stage correction pipeline (pipeline.py orchestrates)
│   │   └── incremental.py  live path: processes only the un-committed tail,
│   │                        caching the frozen prefix (see Live-speed design)
│   └── resources/        radiology_prompt.txt (Whisper priming prompt, ~200 terms)
├── ui/          main_window.py · views.py · recording_session.py · dialogs.py
│   ├── postprocess_worker.py  runs the pipeline OFF the UI thread, latest-only
│   ├── release_gate.py   the desktop dialogs every exit asks (rules: features/report_release)
│   ├── training_capture.py  what a recording tells the (opt-in) training collector
│   ├── web_app.py        assembles the FastAPI app (start-up, routers);
│   │                       `python -m src.ui.web_app` (host/port from settings)
│   ├── web/              one router per job, all thin over features/:
│   │                       state.py (the only between-request state: settings,
│   │                       cached engines, the open report's findings; the
│   │                       seam for per-user state later) · schemas.py ·
│   │                       pages.py (page, assets, bootstrap payload) ·
│   │                       preferences.py · report.py (release gate, exports) ·
│   │                       terms.py (marks, suggestions, learning) ·
│   │                       diagnostics.py (/api/runs, /api/debug/events +
│   │                       /perf, which the developer console and the unlisted
│   │                       /developer page read) · dictate.py (/ws/dictate)
│   ├── field_navigation.py  Tab / Shift+Tab select the next / previous
│   │                       template field (keys are settings, so a dictation
│   │                       mic's buttons can send them); app.js does the same
│   ├── term_popup.py · term_marks.py   highlight a word → suggestions; and the
│   │                       marks saying which word to highlight. Both are view
│   │                       overlays (ExtraSelection / an underlay div): never
│   │                       text, so no mark can reach an exported report
│   ├── finding_marks.py  the critical-findings gutter: a strip beside the
│   │                       editor, one mark per finding, click to jump to it.
│   │                       Never draws on the radiologist's characters: a
│   │                       finding is a statement about the report, not about
│   │                       a word (contrast term_marks.py, which underlines).
│   │                       Marks sit by position in the DOCUMENT, not by where
│   │                       the text is scrolled, so a finding further down the
│   │                       report still has a mark to click. The web app draws
│   │                       the same strip and count from the same
│   │                       features/report_release.OutstandingFindings, the
│   │                       rules live there, both front-ends only draw
│   ├── tokens.json       the ONLY place a UI colour is written down
│   ├── theme.py          the only reader of tokens.json, renders the Qt sheet
│   │                       and the web page's CSS custom properties, so the two
│   │                       front-ends cannot drift apart
│   ├── styles.py · styles/app.qss · frontends/   desktop + web assets
│   │                       (neither stylesheet contains a hex value;
│   │                        keep it that way, colour has one home)
│   └── __main__.py       enables `python -m src.ui`
├── medical/     critical_findings.py (NegEx) · macros.py ·
│   ├── medical_dict.py   the two wordlists, plus is_english_word, the single
│   │                       "is this a real word?" answer the fuzzy corrector and
│   │                       the marking scan both ask, so they cannot disagree
│   ├── term_lookup.py    highlight a word → what it might have been + what goes
│   │                       with it (lookup); and which words to highlight at all
│   │                       (suspect_terms, three gates, the last of which
│   │                       guarantees every mark has something to offer)
│   └── deid.py           DeIdentifier + PrivacyError, PHI de-id, the upload
│                           safety gate; lives here so dictation/training/cloud
│                           all import it downward without dictation touching
│                           src.cloud.*
├── features/    adaptive_learning.py · audit_log.py · ai_cleanup.py (the
│                  optional Groq polish: the one network feature here, kept out
│                  of dictation/, which stays offline)
│   ├── template_fields.py  what a template field is ([Right / Left], [X]),
│   │                       declared once: the release gate, Tab navigation and
│   │                       pick-list answers ("left") all read it; the web page
│   │                       gets the pattern in its bootstrap payload
│   ├── report_release.py   the gate on every exit (Copy/Save/Word): report
│   │                       checks first (can go back), then critical findings
│   │                       (never withheld; records who was told, how, when)
│   ├── report_checks.py    side / sex / missing-impression checks, scored on
│   │                       scripts/eval/report_checks_corpus.py
│   ├── run_log.py        one capped JSONL record per dictation (timings +
│   │                       optional report text), the history behind /developer
│   └── report_manager.py
├── imaging/     OPTIONAL local chest X-ray assistant (offline inference)
│   ├── classifier.py     TorchXRayVision DenseNet121 wrapper (lazy torch)
│   ├── abstention.py     calibrated rejection gate, the imaging safety gate
│   ├── localization.py   hand-rolled Grad-CAM → region + overlay PNG
│   ├── analyzer.py       orchestrator: classify → abstain → localize → result
│   ├── schemas.py        ImagingFinding / ImagingResult / DISCLAIMER ·
│   │                      ReferenceCase / RetrievalMatch (similar-case retrieval)
│   ├── retrieval.py      orchestration: build/cache reference index, attribute-
│   │                      aware "find similar prior cases" for a query film
│   ├── retrieval_embed.py  EmbeddingExtractor (DenseNet embeddings, lazy torch)
│   ├── retrieval_index.py  EmbeddingIndex, HNSW (hnswlib) / numpy search + I/O
│   ├── datasets.py       local labelled-image dataset loading for fine-tune/eval
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
│   └── exceptions.py     CloudError hierarchy (+ ImagingError, GroqError);
│                          re-exports PrivacyError from medical/deid.py
├── devtools/    speech_test.py: the "Test voice" in the developer console.
│                  Types a report, Windows speaks it into dictation in place of
│                  the mic. Removable: this folder, /api/dev/speak, and the
│                  speechTest block in app.js + form in app.html
├── training/    collector.py · schemas.py · staging_db.py (SQLite)
├── templates/   plain-text report templates (RSNA / MSK / generic)
└── resources/   medical_terms.txt (broad generic wordlist, membership net) ·
                  radiology_lexicon.txt (curated radiology terms, the clean
                  spelling-correction snap targets)
```

Other optional, off-by-default add-ons:
- `features/ai_cleanup.py`: on-demand Groq report polish (NOT in
  the per-chunk pipeline); de-identifies first, key in keychain, consent-gated.
- `scripts/lightning/`: training entrypoints run ON Lightning AI: `train_whisper`
  + `convert_to_ct2` (voice), `train_text_corrector`, `train_scan_classifier`.
  Their dependencies are the `train` extra in `pyproject.toml`.

## Dictation data-flow (always local)

```
desktop:  microphone → audio.py → growing WAV → worker.py (QThread adapter)
web:      microphone (raw: browser noise suppression, auto-gain, echo
          cancellation all off) → AudioWorklet (resampler.js only where the
          browser cannot run at 16 kHz) → /ws/dictate (16-bit PCM @16k)
both:     → stream/live_session.py (one loop: Deepgram's live socket when it
            streams, else chunk-once over an in-memory buffer)
          → asr/ (AsrEngine port → Parakeet, Whisper behind it) → postprocess/ (10 stages)
          → UI (views.py / web_app.py) → report_manager.py (.docx / .txt export)
```

**Dictation goes in at the caret, not at the end.** The dictated region has
two ends (desktop: two `QTextCursor`s, the start fixed on insert and the end
growing; web: the text before and after the caret), and text after the region,
the rest of a template, is never rewritten. A selected template field is
replaced; if the dictation is exactly one of a pick-list's options it becomes
that option (`template_fields.resolve_choice`, run by the desktop renderer and
by the web server). Saying "next field" mid-recording is not built: it would
split the region inside the live loop; Tab (or a mic button) does it between
dictations.

Both front-ends use two models: `live_model_size` (fast) decodes what appears
while you speak, and `model_size` re-decodes the low-confidence chunks after
Stop. Both sizes are Whisper's: on the default `local` engine Parakeet answers
in both slots, so the polish is skipped (rule 8 below). **Stop never waits on that second pass in either front-end**: the live
text is handed back at once and the polish upgrades it in the background.
Which makes one rule load-bearing, and it is written in both places: if the
report has been edited since it was handed over, the polished version is
dropped rather than applied. Overwriting a clinical report someone has already
corrected is the worst outcome the feature could have. Because Record comes
back before the polish ends, a second recording can start on top of an
unfinished one: the desktop disconnects the old worker's signals before the new
session exists (`recording_session._abandon_unfinished_session`), so a stale
pass cannot reach the new document.

**A dictation owns the report until it is handed back.** In the browser that
is `sessionPhase` (idle → starting → recording → finishing → idle): Record
works only from idle, the editor is read-only, and New, Clear, a template, a
phrase, undo and every exit (Copy/Save/Word) refuse until the hand-back
(`stopped`) arrives. A whole-report write (New, Clear, template) goes through
`replaceReport`: it asks first, stays one undo step away, and gives the page a
new `reportId`. The server keeps findings answers per report id
(`web/state.findings_for`), so an acknowledgement never answers for the next
patient. Both front-ends ask about, write and audit only the findings still
outstanding (`OutstandingFindings.pending`).

**Files the two apps share are changed under a cross-process lock**
(`core/file_lock.py`, `json_store.update_json`): settings, the model
registry and learned corrections each re-read the file and apply only their
own change. Temp audio lives in one folder per running app
(`paths.temp_dir`), and start-up clears only folders whose app has exited.

Front-end files are cached in memory keyed on the file's modification time
(`web/pages._frontend_cache`), and every page and asset is served `no-store`. Both
halves matter: without the mtime key an edit to app.js needed a server restart,
and without `no-store` the browser kept running the previous release's script
against the current server: which is how the page went on using the old
record-then-upload path, and felt many seconds slower, long after live dictation
had landed.

## Live-speed design (why dictation keeps up)

The live worker re-emits the *whole* transcript every cycle. Four rules keep the
per-cycle cost flat instead of growing with the length of the report: a long
dictation used to get slower the longer it ran:

1. **The pipeline never runs on the UI thread.** `ui/postprocess_worker.py` owns
   a `QThread` with a one-slot mailbox: only the *latest* transcript is
   processed (older ones are stale by definition), and each result carries a
   sequence number so a late pass can't overwrite a newer one. Worker signals
   are connected to **bound `MainWindow` slots, never lambdas**: a signal
   connected to a plain callable has no receiver thread affinity, so Qt would
   run it in the *emitting* thread and mutate the editor off the UI thread.
2. **Only the un-committed tail is post-processed.** `postprocess/incremental.py`
   caches the processed form of the frozen prefix and splits at a *sentence
   boundary at or before the commit frontier*: so every piece the pipeline sees
   starts where a sentence starts, exactly as it would inside the full document.
   No safe boundary yet → it falls back to whole-document processing. The final
   pass after recording stops always reprocesses the whole document, so the
   report the radiologist reviews is never a partially-processed artefact.
   That final pass goes through the same worker thread, submitted with the
   highest sequence number of the session, so Stop returns immediately and the
   full-document result is still guaranteed to be the last text applied.
3. **Each chunk is decoded exactly once.** `dictation/stream/` finds VAD silence
   boundaries in the still-open tail (`vad.py`), turns them into chunk cuts
   (`segmenter.py`), and permanently freezes each closed chunk's decode
   (`ledger.py`): nothing ever re-decodes committed audio. Only the still-open
   tail (bounded by `ChunkPolicy.force_cut_sec`, default 20s) is re-decoded
   cycle to cycle, purely for a stable live preview via LocalAgreement-2
   (`tail.py`). **The first decode is shown at once and may be corrected
   exactly once; after that the preview never takes a word back.** Text that
   un-writes itself mid-sentence reads as the app losing the dictation, so
   confirmed words are pinned until the chunk closes. But requiring agreement
   before showing anything means the earliest words possible are the SECOND
   decode, and a decode costs over a second whatever it is handed: measured end
   to end in the browser, that put the first words of a dictation 13.8 seconds
   after the button was pressed. The first decode is therefore shown
   provisionally, the next decode may revise it, and the pin applies from then
   on. One bounded correction at the very start is not the failure this rule
   exists to prevent. **Corrected 2026-09-02, read this before trusting the paragraph below.** The flat-cost premise holds only while the cores are idle. Measured again on a loaded machine: wall time still looks flat (0.98s at 1s of audio, 1.47s at 25s) but CPU-seconds double (6.81 to 13.80), because the encoder is flat at a 30s pad while the decoder is linear in tokens. On a busy box the linear half surfaces as wall time and the throttle switches previews off on long tails, which is the failure it exists to prevent. Full measurements and what else turned out false: [docs/lag-map-2026-09-02.md](docs/lag-map-2026-09-02.md).

**The preview is priced per call, not per second of audio.**
   Measured on this machine, one `transcribe()` on the live model costs about
   the same whatever it is handed, 1.33s for a 3s clip, 1.36s for 6s, 1.52s
   for 25s, because Whisper pads every clip to a 30-second window, so the
   encoder does identical work each time. Two rules follow, and both live in
   `rules.should_skip_preview` and its callers: a preview is dropped when *its
   own last measured cost* exceeds `preview_max_lag_sec`, and a new one never
   starts until as long has passed as the last one took. That caps previews at
   half the wall clock and leaves the other half for the committed chunks,
   which are the text that is kept. Pricing the preview by tail length instead
   (`open_tail_sec * decode_cost`, the version this replaced) switched it off
   part-way through every chunk: exactly the stretch where the radiologist has
   said the most and can see the least. After recording stops
   there is no full re-transcribe: a confidence-targeted polish
   (`worker._run_confidence_targeted_polish`) re-decodes only the committed
   chunks whose mean word confidence (from the `AsrEngine` port's
   `want_word_confidence`) fell below the ceiling, plus whatever was still open.
4. **Only the un-decoded tail is ever read off disk.** `worker._read_audio(from_sample)`
   seeks from the ledger's open-tail frontier; it never re-decodes the entire
   growing WAV.
5. **A cloud engine holds its connection open.** `deepgram_engine._http_client`
   is one `httpx.Client` for the life of the process, built in `preload()`.
   A fresh connection per call put a TLS handshake in front of every decode:
   measured against the live API, 1.29s for a 2s clip and 1.36s for a 6s one,
   against 0.18s and 0.16s over a kept-alive connection. Since rule 3 holds the
   next preview back for as long as the last one took, that handshake alone
   stretched live updates to about five seconds apart and tripped
   `preview_max_lag_sec`, so the report arrived in one lump at Stop. Never call
   `httpx.post` directly from an engine. Each call has its own deadline
   (`deepgram_engine.call_timeout`); a flat 30s timeout froze dictation.
6. **The engine declares its cost; chunk sizes follow it.** `EngineCaps.cost`
   → `stream/policy.py`. `chunk_policy: auto` (default) sizes short chunks for
   Deepgram, 2/5 for Whisper; old installs carry 6/15/20 as literal numbers.
7. **A streaming engine does the live work itself** (`asr_streaming`). Audio
   goes up Deepgram's live socket; settled results go straight into the
   ledger. Any failure falls back to chunks from the ledger frontier.
8. **The polish starts before Stop** (`background_polish`), on a background
   thread behind the frontier; only the live loop writes the ledger. Skipped
   entirely when the accurate and live engines are the same.

Measurements and what is still unverified:
[docs/prd-live-dictation-speed.md](docs/prd-live-dictation-speed.md#delivered-2026-09-27).

`core/perf.py` and the in-app developer console are the evidence for all of
the above. Both are in-process only, read on loopback, and persist nothing, so
they do not weaken the offline invariant. What they show and how to read them:
[src/ui/CLAUDE.md](src/ui/CLAUDE.md).

**The first words arrive in about three seconds, and three things had to
change to get there** (all measured in the browser, not estimated). The Silero
VAD was loading lazily inside the first cycle of the first dictation after a
restart, costing 11 seconds cold: it is a startup warmer now
(`warmup._warm_vad`). The opening cycle spent a whole decode on 0.048s of
audio, because a decode costs the same whatever it is handed, so nothing under
`preview_min_tail_sec` (default 1.0s) is decoded at all. And the agreement rule
above threw the first decode away. Together: 13.8s to 2.1s.

**A long pause ends the sentence.** A silence at or beyond `pause_threshold`
breaks the paragraph, and `stream/ledger.close_sentence` puts a full stop on the
text before it. Whisper decodes each chunk in isolation and never hears the
silence that followed, so without this a dictation reads back as one run-on line
per paragraph and the capitalisation stage has no boundary to work from. A
trailing comma is left alone: the speaker was mid-list.

**Words the decoder guessed at say so.** Any word below
`uncertain_word_confidence` (default 0.6) is collected by
`rules.low_confidence_words`, travels with each socket update, and is drawn as a
faint thin underline in the browser: deliberately fainter than the suspect-term
mark, which is a solid dotted line and *does* have alternatives to offer when
clicked. The confidence-targeted polish after Stop replaces a chunk's flagged
words along with its text, so nothing stays underlined that the accurate model
has since settled.

The 10-stage correction pipeline and the two wordlists behind the fuzzy stage
are described in [src/dictation/postprocess/CLAUDE.md](src/dictation/postprocess/CLAUDE.md),
which loads when work touches that folder.

**Five folders keep their guidance in their own `CLAUDE.md`**: each loads only
when work touches that folder: the front-ends and their diagnostics
(`src/ui/CLAUDE.md`), the correction pipeline
(`src/dictation/postprocess/CLAUDE.md`), cloud fine-tuning (`src/cloud/CLAUDE.md`,
opt-in), the scan assistant (`src/imaging/CLAUDE.md`, opt-in), and the
accuracy-measuring harness (`scripts/eval/CLAUDE.md`). Any change to transcription or
post-processing is judged by that harness's numbers, not by reading a sample
(`stage_attribution.py` scores each post-processing stage on its own;
`rescore.py` re-runs the pipeline on stored decodes in seconds):
read `scripts/eval/CLAUDE.md` and [docs/dictation-accuracy.md](docs/dictation-accuracy.md)
before proposing an accuracy change; several obvious ones are already refuted
there.

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
  (`ai_cleanup.get_groq_key`).
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
python -m pytest -q          # safety tables, properties + streaming rules; CI runs both
scripts/mutation_test.sh     # would the tests notice a bug? (docs/testing.md)
npx pyright src              # type check (optional-dep import warnings expected)

# Accuracy + speed measurement: see scripts/eval/CLAUDE.md

# Optional extras (lazy-imported; core app runs without them):
pip install -e ".[imaging]"     # Scan Assistant (local)
pip install -e ".[parakeet]"    # Parakeet ASR engine (local)
pip install groq                                            # AI Cleanup
```