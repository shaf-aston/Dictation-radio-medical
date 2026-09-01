# Dictation speed — what was measured, and what not to try again

Recorded 2026-07-28, after M2 (chunk-once streaming) shipped.

A multi-agent review read the whole live path looking for anything slow that
M3–M6 did not already own. Seven candidate fixes came out of it. Each was given
to two independent reviewers — one told to refute the speed claim, one told to
find a correctness risk — and **all seven were refuted on measurement**. This
file exists so nobody spends another day rediscovering them.

## Where the time actually goes

Measured end-to-end lag between speech and text on screen: **+9 to +11 seconds**
on this CPU-only machine.

| Term | Cost | Who owns it |
|---|---|---|
| Chunk-close latency | 6–20 s by construction | policy, not inefficiency — `chunk_min_sec` / `chunk_force_cut_sec`; M6 makes it tunable |
| Decode of a committed chunk | ~5 s at ~0.31 real-time | **M3 only** |
| Post-stop confidence-targeted polish | re-decode of weak chunks at `final_beam_size` | M6 knob (now in settings) |
| Everything else | 5–14 ms per pipeline pass | noise |

The post-processing pipeline costs **0.08 % of a cycle** — 5–14 ms against the
~6,200 ms it takes to decode the same 20 s of audio. There is no hidden
pipeline hotspot. The decode-bound diagnosis in `CLAUDE.md` holds under
measurement.

M2's gate is not lying either: measured `stream.decode_ratio` of **1.17× and
1.36×** across two runs, against the ≤1.4× target, and live-loop decode
wall/audio of 0.76×. The loop is not falling behind real time.

## The honest ceiling

The most aggressive thing anyone tried — deleting the open-tail preview decode
entirely, which is stronger than any of the seven proposals — moved lag from
+10.5 s to +9.1 s. **1.4 seconds, about 13 % of the lag, paid for by deleting
the live preview**, because skipping that decode also skips the
`_agreement.update()` call that seeds LocalAgreement-2 the next cycle.

**85–90 % of the remaining lag is decode time plus a deliberate chunking
policy.** A faster engine is the only thing that moves it. That is M3.

## The seven, and why each died

| Proposal | Why it was refuted |
|---|---|
| Skip the open-tail preview decode | Its return value looks discarded but is the LocalAgreement-2 seed; killing it adds a full cycle of lag |
| "`decode_ratio` is blind, so the loop is behind" | Measured 0.64–0.76× — already inside the target |
| Shorten the fixed 2.0 s cycle nap | It is idle slack, not a queue; polling faster *raises* `decode_ratio` |
| Cap the VAD re-scan window | The open region is already bounded by `force_cut_sec`; a 22 s cap never binds (max observed tail 11.6 s) |
| Drop word timestamps to skip DTW | Silently deletes committed words via the 8-second hallucination gate (`transcriber.py:352`) |
| Drop `vad_filter` from the polish pass | Chunks *begin* with the pause they were cut at, so the second pass does real work |
| Cache `_project_root()` | 0.566 ms against a 1–3 s cycle — below the noise floor of what it improves |
| Prefilter the 151-rule terminology table | Its ASCII safety valve is switched off upstream by the degree sign in `measurements.py:47`, making it net *slower* on real MSK reports |

## Update 2026-07-28 — the app was not running the model its settings named

The review above measured `small.en` directly, by loading it in a script. The
*app* was loading something else.

`SUPPORTED_MODELS` listed only multilingual sizes — no `.en` variants — while
`dictation_settings.json` said `small.en`. Nothing matched, and both front-ends
dropped it silently:

- **Desktop:** `views.py` built the picker with `addItems(SUPPORTED_MODELS)` then
  `setCurrentText("small.en")`. On a non-editable `QComboBox` that is a **no-op**
  when no item matches, so the picker stayed on its first entry — `tiny` — and
  `recording_session.py:134` handed `tiny` to the worker.
- **Web:** `if model_size not in SUPPORTED_MODELS: model_size = "base"`.
- Worse, `main_window.py` warmed `small.en` from the *setting* while recording
  used `tiny` from the *combo*, so startup paid to load a model it never used.

### What that cost, measured

`python -m scripts.eval.run_eval --set tts --model <m>` — 30 clips, 10.8 min.
Synthetic audio, so these are vocabulary numbers, not real-acoustics numbers.

| model | WER | **medical-term error** | false-correction | RTF (median) |
|---|---|---|---|---|
| `tiny.en` | 8.4 % | **9.4 %** | 7.3 % | 0.14 |
| `base.en` | 10.1 % | 8.8 % | 3.1 % | 0.21 |
| `small.en` | **5.5 %** | **3.4 %** | 11.1 % | 0.47 |

**Medical-term error rate nearly tripled** on the model the app was actually
running. That is the number that matters here — a report can post a respectable
WER while mangling every anatomical word in it.

Fixed: `.en` variants are selectable, `resolve_model()` is the single place a
model name is validated, and it **logs** when it falls back instead of
reassigning in silence. A test asserts the shipped default is a model that
exists, which is the assertion that would have caught this.

**Note on speed:** `small.en` at RTF 0.47 still decodes about twice as fast as
speech arrives. Lag is not decode throughput — it is the chunk-close policy plus
a fixed ~3.6 s cost per decode call (Whisper pads every clip to a 30 s window).
That fixed cost is why shortening `chunk_min_sec` backfires: halving chunk length
nearly doubles total decode.

## Update 2026-07-28 — 81 % of the radiology dictionary never reaches the decoder

The dictionary *is* already inside the decoder, as `initial_prompt`
(`radiology_prompt.txt`, passed by `worker._build_context_prompt`). But Whisper's
prompt slot is 223 tokens and the file is **1,150 tokens**, so faster-whisper
keeps only the last 223 and discards **927**.

Measured against the installed tokenizer, not inferred from the file's comment.

What survives is the generic MRI/ultrasound physics vocabulary at the end of the
file ("Gadolinium, post-contrast, Hyperechoic…"). What is dropped is the whole
front — **ACL, PCL, supraspinatus, infraspinatus, subscapularis, glenohumeral** —
the musculoskeletal terms. The file's header says it orders "most universal terms
at the END" on purpose, so the truncation is intended; the scale of it is not
obviously intended, and the terms being lost are the specialty ones.

Not fixed here, deliberately: choosing *which* 223 tokens is a domain judgement,
and the only instrument that could prove an improvement is the `own` gold set
(real acoustics, MSK vocabulary). The `tts` set is generic radiology, so it would
score the change as noise. **Record the `own` set first, then this is measurable.**

Adding faster-whisper's `hotwords` argument would not help: `get_prompt()` spends
hotwords and the prompt from the *same* 223-token budget, so they compete.

## Update 2026-09-01 — the desktop was writing live text with the accurate model

The web path had used a fast model for live text since 2026-08-22
(`web_live_model_size`). The desktop had never been given that split: it decoded
every live chunk AND every preview with `model_size` (`small.en`). The setting is
now shared as `live_model_size` and both front-ends read it.

### What one decode call costs here

Measured directly, best of three per length, same clip:

| clip | `tiny.en` | `small.en` |
|---|---|---|
| 3 s | **0.70 s** | 4.17 s |
| 6 s | 0.84 s | 3.77 s |
| 10 s | 2.49 s | 4.24 s |
| 15 s | 1.84 s | 4.81 s |
| 20 s | 1.75 s | 5.60 s |
| 25 s | 2.26 s | 8.09 s |

`small.en` costs ~4 s a call whatever it is handed — a 3-second clip costs more
than the audio it contains. That is why the desktop could never catch up: the
preview skipper switched previews off, and committed chunks queued behind them.

### What the split bought, measured on a real 30 s dictation

Replayed at true speaking pace, timestamping every update the screen would show:

| | live = `small.en` (before) | live = `tiny.en` (after) |
|---|---|---|
| first words on screen | 15.9 s | **4.7 s** |
| updates during the 30 s | 5 | **33** |
| words shown when the audio ended | 22 of 33 | **29 of 33** |

## Update 2026-09-01 — shortening the chunks: still refuted, on new numbers

The earlier refutation rested on `small.en`'s ~4 s fixed call cost, which the
live path no longer pays, so it was re-measured rather than assumed. 18 `tts`
clips, `tiny.en` live + `small.en` polish, same clips both rows:

| policy (min/soft/force) | WER | medical-term error | chunks/clip | decode | RTF |
|---|---|---|---|---|---|
| **6 / 15 / 20 (shipped)** | **6.66 %** | **7.11 %** | 2.2 | 216 s | **0.55** |
| 3 / 6 / 8 | 7.31 % | 7.56 % | 4.2 | 358 s | 0.92 |

**Verdict: keep 6/15/20.** Halving the chunk length raised decode by 66 % and
moved both accuracy metrics the wrong way. RTF 0.92 is the number that settles
it: above 1.0 the machine cannot keep up with live speech at all, so 0.92 leaves
no headroom for a slower machine, a longer report, or anything else running. The
accuracy gap is inside the noise (an 8-clip run had the two the other way round),
but nothing here *pays* for the cost, and the mechanism the original note named —
halving chunk length nearly doubles total decode — is confirmed on the fast model
too.

## Update 2026-09-01 — Stop no longer waits, on either front-end

Verified end to end by driving the shipped code, not a stand-in: the web through
a real WebSocket against a running server, the desktop through the real
`MainWindow` with a file in place of the microphone. Quiet machine, same 30 s clip:

| | web | desktop |
|---|---|---|
| first text on screen | 4.9 s | 6.9 s |
| report handed back after Stop | **+0.0 s** | **+0.0 s** |
| accuracy pass lands (background) | +16.1 s | +17.2 s |

`finalise_sec` in the run log therefore no longer measures a wait anyone sits
through. It is relabelled in `/developer` as background time; read it as headroom.

**One thing the polish did not fix, worth knowing.** On the web run it corrected
`hemothorax` → `pneumothorax` and `cardiomedius inal` → `cardiomediastinal`. On
the desktop run of the *same audio* it left both standing, because the polish
only re-decodes chunks whose mean word confidence fell below
`polish_confidence_ceiling` — and the fast model was confident about a word it
got wrong. That is the same weakness already recorded in
[dictation-accuracy.md](dictation-accuracy.md): Whisper's confidence does not
separate "heard correctly" from "heard wrong". Raising the ceiling re-decodes
more chunks and costs the Stop headroom above; it has not been changed.

## A measurement trap that cost an hour (2026-09-01)

The bench and `tts` clips are **22050 Hz**; the live path is 16 kHz throughout.
Resampling them by picking nearest samples (`audio[indices]`, no anti-alias
filter) aliases them badly enough to send WER from ~10 % to ~62 % and make every
clip look like it was truncated — it reads exactly like a corrupt gold set, and
was very nearly reported as one. `run_eval` never hits this because it hands the
engine a *path* and the library resamples properly. Any harness that feeds arrays
must use `faster_whisper.audio.decode_audio(path, sampling_rate=16000)`.

## Acted on

`perf.log_summary` logged stage timings only and never read `gauges()`, so
`stream.decode_ratio` — the M2 exit metric — was absent from the
end-of-recording log and visible only at `GET /api/debug/perf`. Fixed. This is
measurement hygiene; it buys zero milliseconds.
