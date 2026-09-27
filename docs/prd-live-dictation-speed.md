# PRD: live dictation that keeps up with the radiologist

Status: **proposed, 2026-09-27.** Nothing here has shipped. It builds on
[dictation-speed-review.md](dictation-speed-review.md) (what was measured and
refuted, 2026-07/09) and [lag-map-2026-09-02.md](lag-map-2026-09-02.md) (where
the committed text's lag comes from). Read both first; this file does not
repeat their numbers, it acts on them.

The complaint is "dictation is really slow". This PRD pins that to numbers,
says which module owns each part of the lag, and proposes the module seams
that make the fast path the default rather than a tuning exercise.

---

## 1. The finding that changes the plan

Every earlier speed review assumed the decoder was local Whisper, where one
`transcribe()` costs about 1 to 4 s whatever it is handed (the 30 s pad). The
chunk policy, the preview throttle, the polish batching and LocalAgreement-2
were all designed around that price.

**Since 2026-09-05 the default engine is Deepgram** (`asr/factory.py`,
`DEFAULT_ENGINE = "deepgram"`), measured at **0.16 to 0.18 s per call** over a
kept-alive connection. The pipeline around it did not change. So with the
default engine, today:

| Step | What it costs now | Why |
|---|---|---|
| Decode a committed chunk | ~0.2 s | Deepgram REST, warm connection |
| **Wait for a chunk to close** | **6 to 15 s, by construction** | `ChunkPolicy` shipped defaults are 6 / 15 / 20 (`core/settings.py`), sized to amortise Whisper's per-call cost. Deepgram has no such cost to amortise. |
| Preview of the open tail | ~0.2 s, but only the LocalAgreement prefix is shown | Needs two agreeing decodes of a growing clip, re-sent from the start each time |
| Polish after Stop | re-sends weak chunks to **the same Deepgram model** | `web_app._get_engine(final_model)` builds the *default* chain for the "accurate" model too, so live and final are both `nova-2-medical`. Same audio, same model, same answer: cost with no benefit. |
| Network stall | **up to 30 s** before the chain falls back | `deepgram_engine._TIMEOUT = 30.0` covers connect and read; a hung socket blocks the cycle, and `cycle_task` blocks the next one |
| VAD per cycle | 3 ms (2 s tail) to 32 ms (20 s tail) | measured 2026-09-27 on a 4-core cloud container. **Not a hotspot.** |
| Post-processing per cycle | 5 to 14 ms | measured 2026-07-28. **Not a hotspot.** |

**The dominant cost of "slow" on the default engine is policy, not compute.**
Committed text trails the microphone by the chunk length because the chunk
length was chosen for a different engine. A faster engine was bought, and the
architecture kept charging the slow engine's price.

Three smaller structural findings:

1. **The desktop runs a second copy of the loop.** `worker.LiveTranscribeWorker._run_cycle`
   re-implements `LiveSession.cycle` over a growing WAV, and has already drifted:
   it has no commit-cycle preview skip, calls `build_context_prompt()` per chunk,
   and sleeps up to 2 s between cycles. Every speed fix lands twice or not at all.
2. **Shipped defaults and the tuned settings disagree.** `stream/polish.py` notes
   the eval's best policy is 2 / 5, which lives only in a local settings file.
   A fresh install gets 6 / 15 / 20.
3. **The engine does not tell the pipeline what it costs.** `EngineCaps` carries
   `word_confidence` and `hotwords`. Nothing says "a call costs 0.2 s flat" or
   "I stream natively", so every policy is hard-coded for Whisper.

---

## 2. Goals and non-goals

### Goals (service levels, measured by `scripts/eval/replay.py --realtime` and the socket's `behind_sec`)

| Metric | Cloud engine (Deepgram) | Local engine (Whisper / Parakeet, CPU) | Today |
|---|---|---|---|
| First word on screen after speech starts | **≤ 0.8 s** p50 | ≤ 2.5 s p50 | ~2 to 5 s |
| Committed (kept) text behind the microphone | **≤ 1.5 s** p50, ≤ 3 s p95 | ≤ 5 s p50, ≤ 9 s p95 | 9 to 11 s (local), 6 to 15 s (cloud, by policy) |
| Report handed back after Stop | 0 s (unchanged) | 0 s (unchanged) | 0 s |
| Final text settled after Stop | ≤ 2 s | ≤ 10 s for a 60 s dictation | ~16 s |
| Failover to local on a dead network | ≤ 2.5 s stall, once | n/a | up to 30 s, per call |
| `stream.decode_ratio` (decode wall / audio) | ≤ 0.3 | ≤ 0.7 | 0.55 to 0.76 |

### Accuracy veto (non-negotiable)

No speed change ships if **medical-term error rate** rises on the eval harness
(`scripts/eval/run_eval.py`, `replay.py`) against a baseline report. WER alone is
not enough: a report can post a fine WER while mangling every anatomical word.
Real-acoustics sign-off still needs the `own` gold set recorded
(`python -m scripts.eval.build_sets --set own --record`).

### Non-goals

- Changing the correction pipeline's stages or wordlists.
- GPU support. Welcome, but the targets above are for a CPU laptop.
- Removing the offline path. Local engines stay first-class; they get their own
  policy instead of sharing the cloud engine's.
- Any change to the invariants in `CLAUDE.md` (edit-wins-over-polish, PHI, keys
  in keychain, Stop never waits).

---

## 3. Who and what

A radiologist dictating a 30 s to 3 min report, reading a film while talking,
pausing mid-sentence to look. What they judge "slow" by, in order:

1. **Words I said a moment ago are not on screen** (committed + preview lag).
2. **Words on screen changed after I read them** (preview instability).
3. **I pressed Stop and the text is incomplete or still changing** (already solved: Stop hands back at once; keep it).
4. **It froze** (network stall, "Catching up" when the machine is not actually behind).

---

## 4. Target architecture: seams and modules

The design principle: **the engine declares its shape and price; the stream
layer picks a strategy from that declaration; the front-ends only feed audio
and draw updates.** Today the price is implicit in constants tuned for Whisper.

```
                 ┌──────────────── front-ends: feed PCM, draw LiveUpdate ────────────────┐
                 │  web_app /ws/dictate (asyncio)        desktop worker (thin Qt adapter) │
                 └───────────────────────────────┬────────────────────────────────────────┘
                                                 │ feed(pcm) / updates()
                              stream/live_session.py  LiveSession (orchestrator)
                              owns: audio buffer · ChunkLedger · IncrementalPostprocessor
                                                 │
                         picks from EngineCaps   ▼
             ┌───────────────────────────────────┴──────────────────────────────┐
   stream/strategies/chunked.py                                 stream/strategies/streaming.py
   (Whisper, Parakeet: batch engines)                           (Deepgram live: streaming engines)
   VAD → segmenter → decode closed chunk once                   push PCM → interim → preview
   open tail → LocalAgreement-2 preview                         final/endpoint → ledger.commit
   ChunkPolicy from caps.cost                                   no VAD, no segmenter, no re-decode
             └───────────────────────────────────┬──────────────────────────────┘
                                                 ▼
                                asr/ port: BatchAsrEngine · StreamingAsrEngine
                                           EngineCaps{cost, streaming, network, ...}
                                asr/engines/: deepgram (batch + live) · parakeet · faster_whisper
                                asr/resilience.py: deadlines · circuit breaker · failover
```

### 4.1 `asr/`: the engine tells the truth about itself

`asr/types.py::EngineCaps` gains three fields. Everything downstream reads
these instead of assuming Whisper:

```python
@dataclass(frozen=True)
class CostModel:
    fixed_sec: float        # per-call floor (Whisper ~1.0-4.0, Deepgram ~0.18)
    per_audio_sec: float    # marginal cost per second of audio (Whisper decoder is linear in tokens)

@dataclass(frozen=True)
class EngineCaps:
    word_confidence: bool
    hotwords: bool
    cost: CostModel
    streaming: bool         # implements StreamingAsrEngine
    network: bool           # audio leaves the device (drives UI badge + failover policy)
```

`asr/port.py` keeps `AsrEngine` (renamed `BatchAsrEngine`, alias kept) and adds:

```python
class StreamingAsrEngine(Protocol):
    def open_stream(self, ctx: TranscribeContext) -> "AsrStream": ...

class AsrStream(Protocol):
    def push(self, pcm16: bytes) -> None: ...            # non-blocking
    def events(self) -> Iterator[StreamEvent]: ...       # Interim | Final | SpeechEnded | Error
    def finish(self, timeout: float) -> None: ...        # flush on Stop
```

`Final` carries words with absolute times, so it maps straight onto a ledger
chunk. Deepgram's live API provides interim results, endpointing and per-word
confidence on one socket, which is exactly what chunk-once + LocalAgreement-2
reconstruct by hand for Whisper.

`asr/resilience.py` (new) takes failover policy out of `ChainEngine`:

- **Deadlines per call**, not one 30 s timeout: connect 1.5 s, read
  `max(2.0, 3 × expected)` where *expected* comes from `CostModel`.
- **Circuit breaker** per provider: after 2 consecutive transient failures,
  skip it for 30 s, then let one call probe it. Today only auth failures are
  remembered (`_DEAD_PROVIDERS`); a network outage is re-paid on every decode.
- **Streaming failover**: on a dropped socket, the unacknowledged audio is still
  in `LiveSession`'s buffer, so the chunked strategy resumes from the ledger's
  open frontier on the local engine. Nothing is lost; the preview just gets slower.

`asr/factory.py` stays the only name-to-engine map, and gains
`create_pair(settings) -> (live, final)`, so "which engine re-decodes after Stop"
is decided in one place. **Rule: if `final` would be the same model as `live`,
there is no polish pass.** With Deepgram live, `final` is local `small.en` only
if the eval shows it beats `nova-2-medical` on medical-term error; otherwise
polish is off for the cloud path.

### 4.2 `stream/`: one orchestrator, two strategies

`LiveSession` keeps what it owns today (buffer, ledger, postprocessor,
`LiveUpdate`, `close_open_tail_fast`, `finalize`) and delegates "how does text
get into the ledger" to a strategy chosen from `EngineCaps.streaming`.

| Module | Responsibility | Status |
|---|---|---|
| `stream/live_session.py` | buffer, ledger, postprocess, updates, Stop | exists; slimmed |
| `stream/strategies/chunked.py` | today's `cycle()` body: VAD → cut → decode once → LA-2 preview | move, don't rewrite |
| `stream/strategies/streaming.py` | push PCM; `Interim` → preview; `Final` → `ledger.commit` | new |
| `stream/policy.py` | `policy_for(caps) -> ChunkPolicy`, preview pacing from `CostModel` | new; replaces constants |
| `stream/ledger.py` | the one frozen record, both strategies write to it | unchanged contract |
| `stream/polish.py` | after Stop; skipped when `final is live` | small change |
| `stream/segmenter.py`, `vad.py`, `tail.py` | used by `chunked` only | unchanged |

**Why the ledger stays the single source of truth for both:** the sentence
close on a long pause, the uncertain-word marks, the ribbon's `savedSec`, the
polish's chunk selection and the edit-wins rule all read the ledger. A
streaming engine that bypassed it would need all five re-implemented.

`policy_for(caps)` replaces the shipped 6 / 15 / 20 for every engine:

```
fixed_sec ≥ 0.8  (local Whisper)  → min 2.0, soft 5.0, force 12.0   (eval-measured best, now the shipped default)
fixed_sec < 0.5  (cloud batch)    → min 0.8, soft 3.0, force 8.0    (commit at the first real pause)
streaming                          → no ChunkPolicy; engine endpointing decides
```

These three rows are starting points, each to be settled by `replay.py`
against the accuracy veto, not accepted from this table.

### 4.3 Two lanes, not one thread (chunked strategy only)

Lag-map lever 5: the commit of a closed chunk queues behind the previous
cycle's preview on the same worker thread (~1 to 3 s on Whisper). Split
`cycle()` into:

- **Commit lane**: the only writer to `_ledger`, `_uncertain`, `_post`. Never waits on a preview.
- **Preview lane**: reads an immutable snapshot `(buffer view, open_start, marks)`,
  returns text, is **droppable** (a newer snapshot cancels the older result).
  Writes only `_agreement` / `_last_stable`, which it owns.

On Whisper this needs `num_workers=2` or a second model instance at half the
threads (measured 6.82 s → 4.05 s median commit lag). On Deepgram-batch the
decode is I/O, so the two lanes are simply two threads. The lock problem the
lag map names disappears by construction: each lane owns disjoint state.

### 4.4 Front-ends: adapters only

- **Web** (`ui/web_app.py /ws/dictate`): already feeds `LiveSession`. Change:
  build it through `factory.create_pair` and `policy_for`, and stop reading
  chunk knobs directly.
- **Desktop** (`dictation/worker.py`): delete `_run_cycle` / `_decode_open_tail` /
  the growing-WAV poll. The recorder's callback calls `session.feed(block)`
  (the WAV is still written for autosave, not read back). The worker becomes a
  QThread that runs the session's lanes and re-emits `LiveUpdate` as Qt
  signals, bound to `MainWindow` slots per the existing rule. One loop, two skins.

### 4.5 What does not move

`postprocess/` stays committed-text-only and off the UI thread. VAD stays
Silero. `warmup.py` keeps warming every tier of the chain (a failover must not
pay a cold load). `core/perf.py` + `event_log.py` remain the evidence and gain
the stage names below.

---

## 5. The microsteps, with a budget each

What one piece of speech passes through on its way to committed text, with the
budget this PRD sets and who owns it. A step that blows its budget shows up in
the developer console under its stage name.

| # | Step | Owner | Budget (cloud) | Budget (local) | Stage name |
|---|---|---|---|---|---|
| 1 | Mic → AudioWorklet → socket frame | `frontends/app.js` | ≤ 100 ms frames | same | `web.frame` (new) |
| 2 | `feed()` append | `LiveSession.feed` | < 1 ms | same | - |
| 3 | Cycle scheduling gap | `web_app` / desktop adapter | ≤ `live_cycle_sec` 0.5 s | same | `web.cycle` |
| 4 | VAD over open tail | `stream/vad.py` | n/a (streaming) | ≤ 40 ms | `stream.vad` (new) |
| 5 | Chunk close decision | `segmenter` via `policy_for` | endpointing ≤ 0.5 s after pause | ≤ `trailing_silence_sec` 0.6 s | - |
| 6 | Commit decode | engine | ≤ 0.3 s | ≤ 1.5 s (`tiny.en`, 8 threads) | `stream.live.chunk` |
| 7 | Word confidence | engine | free (Deepgram returns it) | +0.45 to 0.6 s: keep on commit only | - |
| 8 | Ledger commit + sentence close | `ledger.py` | < 1 ms | same | - |
| 9 | Post-process committed tail | `incremental.py` | ≤ 15 ms | same | `postprocess.incremental` |
| 10 | Preview decode | engine | interim, ≤ 0.3 s | ≤ `preview_max_lag_sec` | `stream.live.preview` |
| 11 | JSON update → DOM | `web_app.send` / `app.js` | ≤ 20 ms | same | - |
| 12 | Failover on error | `asr/resilience.py` | ≤ 2.5 s once, then breaker | n/a | `asr.failover` (new) |
| 13 | Stop → hand-back | `close_open_tail_fast` | ≤ 0.3 s (stream flush) | ≤ 1.5 s | `stop.tail` |
| 14 | Polish after Stop | `polish.py` | skipped when `final is live` | ≤ 10 s / 60 s audio | `stream.polish.*` |

Steps 4, 6 and 10 are where the local path's time goes; step 5 is where the
cloud path's time goes today.

---

## 6. Delivery plan

Each phase is independently shippable, lands on its own branch, and is judged
by `replay.py --realtime` + `run_eval.py` against a baseline taken first.

### P0: stop paying Whisper's price on the cloud engine (hours, low risk)

1. `deepgram_engine`: `httpx.Timeout(connect=1.5, read=max(2.0, 1.0 + 0.3 × clip_sec))`
   per call instead of a flat 30 s. Transient failures count toward a breaker in `ChainEngine`.
2. `factory.create_pair`: skip the polish when live and final resolve to the same
   provider/model. Removes pointless post-Stop Deepgram calls.
3. Chunk policy chosen from the primary engine (Deepgram-batch row in 4.2),
   and the Whisper row's 2 / 5 made the **shipped** default instead of a local
   settings override.
4. Baseline and after: `behind_sec` p50/p95, first-word ms, medical-term error.

**Exit:** cloud committed lag p50 ≤ 3 s; no medical-term regression.

### P1: `EngineCaps.cost` / `streaming` / `network` + `stream/policy.py` (a day)

Pure refactor of P0's special-cases into the declared-capability seam. Tests:
`policy_for` table, each engine's caps, and the shipped-default assertion
(the test that would have caught the `small.en` / `tiny` drift, extended to chunk policy).

### P2: Deepgram live streaming strategy (days, the big win for the default path)

`DeepgramEngine.open_stream` over `wss://api.deepgram.com/v1/listen` with
`interim_results`, `endpointing`, `utterance_end_ms`, `nova-2-medical`, keywords.
`strategies/streaming.py` maps `Final` → `ledger.commit`, `Interim` → preview.
Failover to the chunked strategy from the ledger frontier on socket loss.

**Exit:** first word ≤ 0.8 s, committed lag p50 ≤ 1.5 s on the cloud path;
socket-kill test shows no lost words and ≤ 2.5 s stall.

### P3: desktop onto `LiveSession` (a day or two)

Delete the duplicate loop. **Exit:** desktop and web produce identical ledgers
for the same replayed audio (a new test drives both).

### P4: two lanes for the chunked strategy (days; needs P1)

Commit lane / droppable preview lane, `num_workers=2`. **Exit:** local committed
lag p50 ≤ 5 s on the `tts` set at `--realtime`; `decode_ratio` ≤ 0.7.

### P5: background accurate pass behind the frontier (later; needs P4)

Lag-map lever 6: re-decode weak chunks with the accurate model one or two chunks
behind the frontier instead of only after Stop, so "final settled" approaches 0.
Must obey the edit-wins rule per chunk, not per report.

### Also worth evaluating (measure, don't assume)

- `nova-3-medical` with key-term prompting in place of `nova-2-medical` + `keywords`.
  Compare on medical-term error before switching.
- Parakeet as the live local engine: CTC has no 30 s pad, so its `CostModel`
  may put it in the cheap row and give the offline path cloud-like lag.
- `BatchedInferencePipeline` for the polish only (27 to 36 % faster, one input).

---

## 7. What not to do again

Already refuted on measurement, see [dictation-speed-review.md](dictation-speed-review.md):
deleting the preview decode, shortening the cycle nap, capping the VAD window,
dropping word timestamps on commit, dropping `vad_filter` from polish, caching
`_project_root()`, prefiltering the terminology table, and **shortening chunks on
Whisper below 2 s** (decode total rises faster than lag falls). Two concurrent
decodes on one `num_workers=1` model buy nothing.

Added by this review: **VAD per cycle is 3 to 32 ms** and post-processing is
5 to 14 ms. Neither is worth optimising before steps 5, 6 and 10.

---

## 8. Risks and open questions

| Risk | Mitigation |
|---|---|
| Short cloud chunks lose context and hurt medical terms | `replay.py` on every policy row; the accuracy veto decides, not this table |
| Streaming socket adds a second connection lifecycle | Contained in `strategies/streaming.py`; the batch Deepgram path stays as the fallback tier |
| Deepgram is a PHI egress point (audio cannot be de-identified) | Unchanged from today's product decision; `EngineCaps.network` drives a visible "cloud" badge. BAA remains a compliance question, not an engineering one |
| Two-lane lock bugs | Disjoint ownership (4.3) plus a replay test that runs lanes under a stressed CPU |
| Synthetic eval set flatters confidence | Record the `own` set before P2/P4 exit sign-off |

**Open question for the owner:** should the post-Stop polish on the cloud path
be a local `small.en` pass at all? It only earns its place if the eval shows it
beats `nova-2-medical` on medical-term error. If it does not, the cloud path's
"final settled" target becomes "at Stop".
