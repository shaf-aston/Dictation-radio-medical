# Module map

What every file in `src/` is for. Moved out of the root `CLAUDE.md` so it is
read when needed instead of on every request.

```
src/
├── core/        settings.py (JSON at project root) · logging_setup.py ·
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
│   │                       engines/parakeet_engine.py (onnx-asr; no prompt,
│   │                       one model shared process-wide; full precision
│   │                       unless parakeet_precision says int8; the
│   │                       parakeet_model setting picks stock "standard" or
│   │                       a fine-tune folder under data/models/fine_tuned/,
│   │                       "Voice model" in both front-ends, warmed on
│   │                       switch by warmup.rewarm) · prompt.py
│   │                       (the radiology priming vocabulary, same for every
│   │                       engine) · models.py (known model names + fallback)
│   ├── stream/             chunk-once streaming, vad.py (Silero VAD, bundled
│   │                       with faster-whisper, no new dep) · segmenter.py
│   │                       (pure VAD-marks→chunk-cuts policy) · policy.py
│   │                       (chunk sizes from the engine's cost) · live_session.py
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
│   │                        caching the frozen prefix (see src/dictation/CLAUDE.md)
│   └── resources/        radiology_prompt.txt (Whisper priming prompt, ~200 terms)
├── ui/          main_window.py · views.py · recording_session.py · dialogs.py
│   ├── postprocess_worker.py  runs the pipeline OFF the UI thread, latest-only
│   ├── web_app.py        FastAPI single-page app (host/port from settings),
│   │                       plus /developer, the local diagnostics table of
│   │                       recorded runs (unlisted; no link from the report UI),
│   │                       and GET /api/debug/events + /api/debug/perf, which
│   │                       are what the in-page developer console reads
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
├── features/    accent_corrections.py · adaptive_learning.py · audit_log.py
│   ├── run_log.py        one capped JSONL record per dictation (timings +
│   │                       optional report text), the history behind /developer
│   └── file_manager.py · report_manager.py · report_analyzer.py
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
- `dictation/postprocess/llm_cleanup.py`: on-demand Groq report polish (NOT in
  the per-chunk pipeline); de-identifies first, key in keychain, consent-gated.
- `scripts/lightning/`: training entrypoints run ON Lightning AI: `train_whisper`
  + `convert_to_ct2` (voice), `train_text_corrector`, `train_scan_classifier`.
  Their dependencies are the `train` extra in `pyproject.toml`.
