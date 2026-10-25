# src/dictation: data-flow and live-speed design

Loads only when work touches `src/dictation/`. The always-loaded rules for this
project are in the root `CLAUDE.md`.

## Dictation data-flow (always local)

```
desktop:  microphone → audio.py → growing WAV → worker.py (QThread adapter)
web:      microphone (raw: browser noise suppression, auto-gain, echo
          cancellation all off) → AudioWorklet (resampler.js only where the
          browser cannot run at 16 kHz) → /ws/dictate (16-bit PCM @16k)
both:     → stream/live_session.py (one loop: Deepgram's live socket when it
            streams, else chunk-once over an in-memory buffer)
          → asr/ (AsrEngine port → Parakeet, Whisper behind it) → postprocess/ (the stage list)
          → UI (views.py / web/dictate.py) → report_manager.py (.docx / .txt export)
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
in both slots, and its re-decode of a weak chunk together with its
neighbours still corrects words (rule 8 below). **Stop never waits on that second pass in either front-end**: the
still-open tail is decoded by the fast model first, then the complete text is
handed back and the polish upgrades it in the background. Handing back the last
preview instead loses the last words spoken, since the polish that would add
them is dropped once the report is edited.
Which makes one rule load-bearing, and it is written in both places: if the
report has been edited since it was handed over, the polished version is
dropped rather than applied. Overwriting a clinical report someone has already
corrected is the worst outcome the feature could have. Because Record comes
back before the polish ends, a second recording can start on top of an
unfinished one: the desktop disconnects the old worker's signals before the new
session exists (`recording_session._abandon_unfinished_session`), so a stale
pass cannot reach the new document.

## Live-speed design (why dictation keeps up)

The live worker re-emits the *whole* transcript every cycle. Eight rules keep the
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
   exists to prevent. **Corrected 2026-09-02, read this before trusting the paragraph below.** The flat-cost premise holds only while the cores are idle. Measured again on a loaded machine: wall time still looks flat (0.98s at 1s of audio, 1.47s at 25s) but CPU-seconds double (6.81 to 13.80), because the encoder is flat at a 30s pad while the decoder is linear in tokens. On a busy box the linear half surfaces as wall time and the throttle switches previews off on long tails, which is the failure it exists to prevent. Full measurements and what else turned out false: [docs/lag-map-2026-09-02.md](../../docs/lag-map-2026-09-02.md).

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
   (`stream/polish.polish`, run by `LiveSession.finalize`; the desktop calls it
   from `worker._run_final_polish`) re-decodes only the committed
   chunks whose mean word confidence (from the `AsrEngine` port's
   `want_word_confidence`) fell below the ceiling, plus whatever was still open.
4. **Only the un-decoded tail is ever read off disk.** `worker._read_audio(from_sample)`
   seeks past the samples already fed to the LiveSession (`_fed`); it never
   re-reads the whole growing WAV.
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
   → `stream/policy.py`. `chunk_policy: auto` (default) sizes 1/3/8s chunks for a
   cheap (cloud) engine, 2/5/20s for local ones (Parakeet, Whisper); old
   installs carry 6/15/20 as literal numbers.
7. **A streaming engine does the live work itself** (`asr_streaming`). Audio
   goes up Deepgram's live socket; settled results go straight into the
   ledger. Any failure falls back to chunks from the ledger frontier.
8. **The polish starts before Stop** (`background_polish`), on a background
   thread behind the frontier; only the live loop writes the ledger. Skipped
   entirely when the accurate and live engines are the same cloud model,
   which gives the same words again. A local model is re-run: even greedy
   Parakeet decodes a weak chunk differently with its neighbours (measured
   2026-10-25, skipping it cost WER on noisy and real voices).

Measurements and what is still unverified:
[docs/prd-live-dictation-speed.md](../../docs/prd-live-dictation-speed.md#delivered-2026-09-27).

`core/perf.py` and the in-app developer console are the evidence for all of
the above. Both are in-process only, read on loopback, and persist nothing, so
they do not weaken the offline invariant. What they show and how to read them:
[src/ui/CLAUDE.md](../ui/CLAUDE.md).

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

The correction pipeline and the two wordlists behind the fuzzy stage
are described in [postprocess/CLAUDE.md](postprocess/CLAUDE.md).
