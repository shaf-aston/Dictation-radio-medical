# Radio Dictate — Architecture, Flows & Performance

Offline medical dictation workstation. Speech-to-text runs **locally** (Whisper
via `faster-whisper`/CTranslate2); by default nothing leaves the device. Two
front-ends (PySide6 desktop, FastAPI web) share one dictation core.

This document is the **flow + performance** map. The per-module breakdown lives
in [CLAUDE.md](CLAUDE.md) — not duplicated here. Read that for "what each file
is"; read this for "how a dictation moves through the system and where the time
goes."

> **Superseded by the M1/M2 rebuild.** Sections 2 and 3 below describe the
> sliding-window + `commit_lag_sec` tuning pass — since replaced by the
> chunk-once streaming architecture (`src/dictation/asr/` + `stream/`;
> `window_state.py`/`text_diff.py` deleted). They are kept as the historical
> record of that pass's reasoning. **CLAUDE.md's "Live-speed design" section
> is the current, authoritative description**; §2's hot-path table and §3's
> tunable-knobs table below are current only where noted inline.

---

## 1. The dictation user flow (desktop)

```
Open app ─▶ pick template (optional) ─▶ click Record (F5)
   │
   ├─ recorder.start() writes mic audio to a temp WAV          [audio thread]
   ├─ a QThread runs LiveTranscribeWorker                       [worker thread]
   │     chunk-once: VAD-cut closed chunks decoded once and frozen (ledger),
   │     only the still-open tail is re-decoded for a live preview
   │
   ▼ partial(full_transcript) every ~0.4–2 s
on_partial_text (UI thread, cheap) ─▶ PostprocessWorker.submit
   ▶ [own QThread, latest-only, incremental tail-only] postprocess (10 stages)
   ▶ on_processed_text ─▶ replaces the dictated region (cursor-anchored at
     `window._dictation_start`, so a loaded template or an in-progress edit
     never gets overwritten)
   │
Stop (F6) ─▶ worker.finalize() (confidence-targeted re-decode of low-confidence
   chunks + the open tail) ─▶ finished
   │
on_transcription_finished ─▶ optional one-shot AI polish ('hard' only)
   ▶ critical-findings scan ▶ corrections banner ▶ snapshot
   │
Save .txt / Export .docx  (report_manager)
```

**Threading rule that matters for "feels slow":** Whisper transcription runs on
the **worker thread**; the 10-stage postprocess pipeline runs on its own
**`PostprocessWorker` thread** (`src/ui/postprocess_worker.py`), never inline in
`on_partial_text` and never on the GUI thread — `on_partial_text` only hands the
transcript off. Only the un-committed tail is reprocessed per cycle
(`postprocess/incremental.py`), so cost stays flat as the report grows. Network
AI cleanup is **not** in the live path (see §3).

The **web app** records the whole press-to-talk session in the browser and POSTs
once at stop, so it post-processes **once per recording**, not on a timer — it
was never affected by the live-loop costs below.

---

## 2. The transcription hot path (the loop) — historical, see note above

`src/dictation/worker.py :: LiveTranscribeWorker.run()` — once per cycle:

| Step | What | File |
|------|------|------|
| read | decode growing WAV to float32 | `worker._read_audio` |
| gate | skip cycle unless ≥ `_MIN_GROWTH_SEC` (0.5 s) new audio | `worker.run` |
| **window** | slice the audio to send to Whisper | `WindowState.window_start` |
| decode | `faster-whisper` transcribe (beam=2 live / 5 final) | `transcriber.transcribe` |
| commit | freeze settled text behind the live tail | `WindowState.advance_commit` |
| dedup | stitch committed prefix + new chunk, no repeats | `text_diff.trim_committed_tail` |
| emit | `partial(full_text)` if changed | `worker.run` |

The model is loaded **once** per session and cached; `compute_type` already
falls back `int8 → float32` (CPU-friendly). Those were **not** the problem.

---

## 3. Performance: why it was slow, and what changed

Four parallel readers mapped the code; every performance claim was then
**re-verified by an independent skeptic against the source** before any change.
Two root causes dominated and compounded.

### Root causes (verified)

| # | Root cause | Where | Verdict |
|---|-----------|-------|---------|
| **1** | **Whisper re-decoded a fixed ~25 s window every ~0.5 s tick.** The commit frontier was pinned 25 s behind "now", so the live window never shrank — most of every decode was audio already transcribed. Steady-state for the whole dictation, not just startup. | `worker.py:_window` + `_advance_commit` | REAL — ~2–10× redundant decode (scales with hardware speed) |
| **2** | **The full accumulated transcript was re-run through all 10 postprocess stages every cycle**, not just the new text — cost grew with total dictation length (≈ quadratic over a session). | `recording_session.py:on_partial_text` | REAL |
| 3 | **'Hard' AI cleanup (Groq network call) sat inside the per-chunk pipeline** — selecting it fired a network request every live cycle, violating the "AI cleanup is on-demand only" invariant. | `pipeline.py` | REAL |
| 4 | Adaptive-learning stage ran up to 5000 regexes over the text every cycle with **no quick-scan gate** (the accent stage already had one). | `adaptive_learning.py` | REAL |

Secondary, left as-is (documented, not bugs): live `beam_size=2` is a deliberate
quality choice (beam=1 caused repetition); the audio callback's synchronous disk
write is a defensive concern only under a stalling disk.

### Fixes applied this pass

| Fix | Change | Impact | Risk |
|-----|--------|--------|------|
| **#1 window** | Decoupled **commit-lag** from the window ceiling. Text older than `commit_lag_sec` (default **8 s**) is committed, so the live window shrinks to `commit_lag + overlap` ≈ **11 s** instead of a fixed 25 s. `live_window_sec` (25 s) is now only a safety ceiling. | Steady-state decode work cut ~2–3× at defaults; the 0–25 s full-buffer ramp removed. **Lossless by construction**: the window always re-covers the just-committed tail (`commit_lag > overlap`) so `trim_committed_tail` dedups it — proven by `test_window_re_covers_committed_tail_so_no_gap`. | Low–med |
| **#4 learned-corrections gate** | Lower-case the text once; skip any pattern whose trigger word isn't present. | Worst per-stage cost drops from O(5000 × text) to O(few × text); grows no worse as a site accumulates corrections. Output identical (a `\bword\b` regex can only fire if `word` is a substring). | Low |
| **#3 Groq out of live path** | `postprocess_transcript(..., live=True)` skips the AI stage; live calls pass `live=True`. The 'hard' polish now runs **once** on the finished report (`_apply_finished_ai_cleanup`). | No network call per chunk; documented invariant restored. | Low |

### Second pass — transcription-quality fixes (garbage / ",,..," / mumbled words)

A follow-up audit (23 agents, adversarially verified) targeted the *quality*
symptoms rather than speed. Root causes and fixes:

| Fix | Root cause | Change | File |
|-----|-----------|--------|------|
| **VAD re-enabled** | `dictation_settings.json` had `vad_filter: false` — silence/breath was decoded by Whisper, the canonical source of `,,..,` and phrase garbage. | Flipped to `true`. | `dictation_settings.json` |
| **Punctuation-run strip** | `normalize_spaces` *compacted* `", , . . ,"` into `,,..,` and no stage removed it. | Collapse any run of 2+ punctuation marks to the first mark (single spaced marks — the spoken-punctuation output — are left alone). | `postprocess/text_utils.py` |
| **Hallucination-regex bounding** | `^(?:…|you)` prefix-match silently deleted any segment starting with "you/your/young/goodbye…". | Short phrases now match only as a whole segment; long YouTube outros stay prefix-matched. | `transcriber.py` |
| **Full-quality final pass** | Committed text was frozen from the fast beam=2 live pass; the beam=5 finalize only re-covered the last ~11 s. | Finalize now re-transcribes the **entire** recording at beam=5 + `condition_on_previous_text=True` and emits it as authoritative. | `worker.py:_run_final_pass` |
| **Commit-path overlap dedup** | Boundary-spanning segments were raw-joined into the frozen prefix, duplicating words at every window slide. | `advance_commit`/`maybe_bootstrap` route additions through `trim_committed_tail`. | `window_state.py` |
| **Silence gate** | Near-silent windows were still decoded. | Skip a live cycle when chunk RMS < `silence_rms_floor` (0.002). | `worker.py` |
| **Failed-cycle retry** | Growth baseline advanced before transcribe, so a failed cycle's audio was skipped. | Advance baseline only after a successful transcribe. | `worker.py` |
| **Web batch beam** | `/transcribe` hardcoded `beam_size=1` (greedy) on a one-shot batch job. | Uses `beam_size` setting (default 5). | `web_app.py` |
| **CPU threads** | CTranslate2 used its default 4 intra-op threads regardless of core count; fallback jumped int8→float32. | `cpu_threads = max(4, cores-2)`; `int8_float32` added to the fallback chain. | `transcriber.py` |
| **O(1) corrections banner** | `if c not in list` per change per cycle was quadratic over a session. | Companion `set` for membership. | `recording_session.py` |

New knobs: `beam_size` (5) and `silence_rms_floor` (0.002) in settings.
**Still deferred:** incremental live postprocess (§5 item 1) — the dominant cost
(§3 #1) is already fixed, and splitting correction state across the commit
boundary remains medium-high risk.

### Tunable knobs — SUPERSEDED, kept for history only

`commit_lag_sec`/`live_window_sec` no longer exist. The chunk-once rebuild
replaced them with `chunk_min_sec`/`chunk_soft_max_sec`/`chunk_force_cut_sec`
(defaults `6.0`/`15.0`/`20.0`) in `dictation_settings.json` — see
`ChunkPolicy` in `src/dictation/stream/segmenter.py`.

> **Needs a real-audio check.** The streaming change is unit-test-proven lossless
> on synthetic segments, but live dictation quality (word boundaries on fast/run-on
> speech) should be confirmed in the running app before relying on it. If anything
> regresses, raise `commit_lag_sec` toward 25 — instant revert, no redeploy. The
> 'hard' AI-polish finalize path needs a Groq-key + consent smoke test (it is a
> no-op when the feature is off, which is the default).

---

## 4. Modularity & redundancy

The framework-level code (cloud framework/tasks, core, postprocess pipeline,
imaging index) is well-factored — no `dictation → cloud` or `core → UI`
violations. Concrete defects found and **fixed this pass**:

| Issue | Was | Now |
|-------|-----|-----|
| Duplicate `ImagingError` | A second `ImagingError(CloudError)` in `cloud/exceptions.py` shadowed the real `imaging/exceptions.ImagingError(Exception)` — a name-collision trap, never raised/caught. | Deleted; the canonical type is re-exported (mirrors `PrivacyError`). |
| Repeated weights literal | `"densenet121-res224-all"` hardcoded in 3 places across 2 modules. | Single `DEFAULT_SCAN_WEIGHTS` in `imaging/schemas.py`. |
| Private-path access | `adaptive_learning` & `audit_log` reached into `file_manager._data_dir()` + hardcoded filenames, breaking the "I/O goes through path helpers" invariant. | New `learned_corrections_path()` / `audit_log_path()` helpers; both consumers use them. |
| Dead code (YAGNI) | `retrieval.retrieve_similar_images()` + its private `build_from_folder()` — zero callers, rebuilt an uncached index against the file's own caching discipline. | Deleted. |

---

## 5. Deferred — ranked next steps (bigger blast radius)

Not done this pass; each needs its own focused change + review.

1. **Incremental live postprocess (root cause #2, structural).** Have the worker
   emit committed-prefix and live-tail separately; post-process only the tail and
   cache the frozen prefix. Medium-high risk (corrections can cross the commit
   boundary — needs the same overlap-safety as transcription). After fix #1 + #4
   this is no longer the dominant cost, so it is a follow-up, not urgent.
2. **Extract the web frontend.** `web_app.py` is ~2130 lines, ~79 % a hardcoded
   HTML/CSS/JS string. Move it to real files under `src/ui/frontends/` (mirroring
   `ui/styles/*.qss`) so the transport layer is thin and the markup is lintable.
3. **Tokenize remaining UI hardcoding.** Pixel widths, gradient hex stops, the
   `127.0.0.1:8005` bind, status-message timeouts, and the `0.03` mic threshold
   are magic values in UI code → move to qss tokens / a layout-constants module /
   settings, per "config, not constants."

---

## 6. Run & verify

```bash
python -m src.ui            # desktop GUI
python -m src.ui.web_app    # web app on 127.0.0.1:8005
python -m pytest tests/ -q  # all pass; a handful skip on optional deps
ruff check src tests
```

---

## 7. Recent changes (testability & offline)

| Change | What | Why | File(s) |
|--------|------|-----|---------|
| **Window/commit machine extracted** *(superseded — see note at top; `window_state.py` was later deleted by the chunk-once rebuild)* | The pure sliding-window + commit-frontier logic moved out of the Qt worker into a plain `WindowState` class (no PySide6, no I/O). The worker now delegates every window/commit decision to it. | The crown-jewel live-speed arithmetic is now unit-testable without Qt or a Whisper model — `test_worker_logic.py`'s windowing/commit tests run against `WindowState` directly. Also removes the last core→UI coupling in the hot path. | `dictation/window_state.py` (deleted), `dictation/worker.py` |
| **Web front-end de-CDN'd** | The web page pulled Font Awesome from `cdnjs.cloudflare.com`; replaced with a small self-hosted CSS-masked SVG icon set (the same `<i class="fas fa-…">` markup, no network). | The CDN link was a synchronous external request that stalled the page offline and **broke the "nothing leaves the device" invariant** for the web front-end. Now genuinely offline. | `ui/web_app.py` |
