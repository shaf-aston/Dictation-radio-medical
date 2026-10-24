# Audio conditioning: plan and contracts

Cleans microphone audio once, at `LiveSession.feed`, before anything listens to it
(VAD, live decode, polish, Deepgram). Everything is off by default; a step is
switched on only after `scripts/eval/replay.py --condition` shows it passes the
marks below on the degraded set.

## Pass marks (Step 1)

- A kind of bad audio is worth fixing only if it raises medical-term error by 3+ points over clean.
- On each kept kind: medical-term error falls by at least a quarter and at least 2 points; overall WER does not rise.
- Clean audio: no more than 0.3 points worse.
- Speed: `audio.condition` p95 under 0.5 ms per 20 ms browser block, under 3 ms per desktop chunk.
- Latency: words appear no more than 0.05 s later (the chain holds back one hop to one window, 0.016 to 0.032 s; measured 16 to 28 ms with 20 ms browser blocks).
- Determinism: bit-identical output however the audio is split into blocks.

## Files (level 3)

```
src/dictation/conditioning/
  port.py       Step: the one thing a cleaning step must do
  chain.py      Chain: framing, the steps in order, alignment, flush; timed "audio.condition"
  factory.py    build_chain(cfg, sr): the only setting-name -> step mapping
  stages/       highpass.py · denoise.py · expander.py · agc.py · level.py (shared PeakFollower)
```

## Contracts (level 4)

**One transform, steps return gains.** The chain cuts audio into frames (window
`window_samples`, default 512 = 32 ms, hop half that, sqrt-Hann in and out, so
unity gains rebuild the input exactly) and takes one FFT per frame. Each step looks
at the frame's power spectrum and returns a gain; the chain multiplies it in and
hands the result to the next step. One inverse FFT and overlap-add per frame.
Rejected: each step filtering the waveform itself (needs scipy, a new dependency,
and one transform per step). numpy only.

```python
# port.py
class Step(Protocol):
    def gain(self, power: np.ndarray) -> np.ndarray | float:
        """power: |X|^2 of this frame after earlier steps, float64, window//2+1 bins.
        Returns an amplitude gain per bin (array) or for the whole frame (float).
        May keep state from frame to frame; never sees block boundaries."""

# chain.py
class Chain:
    def __init__(self, steps: Sequence[Step], window: int) -> None
    def process(self, block: np.ndarray) -> np.ndarray
        """float32 mono in. Returns the cleaned samples that are ready, aligned to
        the input (sample n out is sample n in, cleaned). Holds back at most one
        window. No steps: returns block unchanged, no transform."""
    def flush(self) -> np.ndarray
        """Stop: returns everything still held. After it, total out == total in."""
    spent: bool   # True after flush, or after a step raised part-way

# Bounds: a NaN/inf sample in becomes 0 (or full scale), so it cannot poison a
# step's state; output is clipped to [-1, 1]. A step that raises spends the
# chain (process/flush then raise RuntimeError): a half-run frame cannot be undone.
# window_samples: an even integer, at most 4096, checked here and only here.

# factory.py
def build_chain(cfg: dict, sr: int) -> Chain
    """cfg is settings["audio_conditioning"]; each step's section is merged over
    get_default's, so an old settings file still gets every knob. ValueError on:
    an unknown section or knob, `on` not true/false, any other knob not a finite
    number, sr not a positive integer. Ranges are checked on steps that are on:
    cutoff_hz in (0, sr/2), min_gain_db < 0, smoothing in [0, 1),
    target_dbfs in [-60, 0), the rest positive."""
```

Fixed order, set in `factory.py`, not a setting: **highpass -> denoise -> expander -> agc**.
Gating before gain, so the volume step never turns up the gaps between words.

| Step | What it does | Knobs |
|---|---|---|
| highpass | gain 0 below `cutoff_hz`, smooth ramp up to it (ramp width follows the cutoff, not a knob); stateless | `cutoff_hz` 80 |
| denoise | tracks each bin's noise floor (falls at once, rises `noise_rise_db_per_sec`, never below 1e-10 so a muted stretch cannot switch it off); stays out until the first non-silent frame; spectral subtraction, never below `min_gain_db` (the floor is the "blend some original back" fix from the research) | `strength` 2.0, `min_gain_db` -20, `noise_rise_db_per_sec` 3, `smoothing` 0.9 |
| expander | follows the main voice's level (`PeakFollower`); frames more than `below_voice_db` under it are turned down, at most `max_cut_db` | `below_voice_db` 6, `max_cut_db` 15, `voice_release_db_per_sec` 6 |
| agc | moves the voice toward `target_dbfs`: turns down at once, up at most `slew_db_per_sec`, never above `max_gain_db`; adapts only on frames within `adapt_within_db` of the voice level, which falls `voice_release_db_per_sec` in a pause, so gaps are never turned up | `target_dbfs` -20, `max_gain_db` 24, `slew_db_per_sec` 20, `voice_release_db_per_sec` 6, `adapt_within_db` 10 |

Settings (`src/core/settings.py` `_DEFAULTS`):

```python
"audio_conditioning": {
    "window_samples": 512,
    "highpass": {"on": False, "cutoff_hz": 80.0},
    "denoise":  {"on": False, "strength": 2.0, "min_gain_db": -20.0, "noise_rise_db_per_sec": 3.0, "smoothing": 0.9},
    "expander": {"on": False, "below_voice_db": 6.0, "max_cut_db": 15.0, "voice_release_db_per_sec": 6.0},
    "agc":      {"on": False, "target_dbfs": -20.0, "max_gain_db": 24.0, "slew_db_per_sec": 20.0,
                 "voice_release_db_per_sec": 6.0, "adapt_within_db": 10.0},
},
```

**LiveSession** gets `conditioner: Optional[Chain] = None`. `feed` appends
`conditioner.process(block)` instead of `block`; whichever of
`close_open_tail_fast` and `finalize` runs first appends `conditioner.flush()`
(`_flush_conditioner`, guarded by `chain.spent`, so it happens once). The append
itself stays one private method, so the raw and cleaned paths cannot differ.
One Chain per session, built at session construction: it learns the room, and a
shared one would carry one dictation's noise into the next.

Callers today: `replay.py` passes one when `--condition` names steps, with the
knobs from settings (only which steps are on comes from the flag);
`simulate_lag.py` passes nothing. Slice 3 adds `worker.py` and `web_app.py`.
Before `web_app.py` feeds a socket message to the chain, it must cap the
message size: the chain trusts its caller on block length.
Alignment matters: the desktop's training collector cuts the raw WAV at the sample
positions the session reports, so cleaned sample n must be raw sample n.

## Build order (Step 3)

Each slice runs and is checked on its own, and lands as its own commit.

0. **Measure first, no app change.** `scripts/eval/build_degraded_set.py` makes
   `tts_quiet` (30 dB down over mic hiss), `tts_noise` (pink noise, 5 dB SNR) and
   `tts_talker` (another report 6 dB below). `replay.py` scores today's app on them
   and on `tts`. A kind under the 3-point mark is dropped, and so is its step.
1. **Skeleton.** `port.py`, `chain.py`, `factory.py` with every step off; the
   settings default; `LiveSession(conditioner=)`; `replay.py --condition`.
   Check: a chain of one unity step gives back the input to float precision, the
   same samples for any block split, and out length == in length after `flush`.
   Pre-mortem: *fails because the output is shifted by a hop*, so the training
   collector cuts the raw WAV in the wrong place. Guard: the sample-for-sample check.
2. **One step per kept kind**, each judged on its own set and on clean `tts`:
   agc for quiet, denoise (+ highpass) for noise, expander for talker.
   Pre-mortem: *the expander cuts the start of a quiet word*. Guard: its release
   is slow and it only turns down, never off (`max_cut_db`).
3. **Wire the app.** `worker.py` and `web_app.py` build the chain from settings;
   a step is switched on by default only if it passed its marks. Check: the
   `audio.condition` p95 in the developer console, and one dictation in the
   browser with the WAV mic.

## Baseline (slice 0, 2026-10-23)

Today's app (Parakeet, no conditioning) through `replay.py` on the 30 `tts` reports.

| Set | WER | Medical-term error | Term error over clean |
|---|---|---|---|
| `tts` (clean) | 6.61% | 7.92% | |
| `tts_quiet` (30 dB down over mic hiss) | 9.39% | 13.51% | +5.6 |
| `tts_noise` (pink noise, 5 dB SNR) | 15.93% | 24.61% | +16.7 |
| `tts_talker` (another report 6 dB below) | 46.67% | 47.43% | +39.5 |
| `tts_talker15` (another report 15 dB below) | 15.22% | 13.27% | +5.4 |

Every kind clears the 3-point mark, so none is dropped. `tts_talker` also breaks
chunking (commit lag p50 6.45 s, one clip produced nothing): a voice that never
stops 6 dB down is beyond what a level-based expander can separate, so the
expander is judged on `tts_talker15`, a colleague across the room. The quiet
set's loss is likely the hiss (about 16 dB under the speech), not the level, since
Parakeet normalises gain: test agc on and off, and denoise, on it.
Reports: `data/eval/reports/cond-baseline-<set>.replay.json`.

## Steps (slice 2, 2026-10-23): none passes, all stay off

Mean over 30 clips, `replay.py --condition <step>`, best attempt per step.

| Step | Set | WER | Term error (baseline) | Verdict |
|---|---|---|---|---|
| agc | `tts_quiet` | 9.41% | 12.79% (13.51) | fails: Parakeet already normalises level |
| denoise | `tts_quiet` | 8.84% | 11.14% (13.51) | fails: -2.4 pts is 17.5%, short of a quarter |
| denoise | `tts_noise` | 15.01% | 22.05% (24.61) | fails: -2.6 pts is 10% |
| expander | `tts_talker15` | 12.00% | 9.92% (13.27) | passes on synthetic speech |
| expander | `tts` (clean) | 6.55% | 7.96% (7.92) | passes |
| all four | `tts` (clean) | 6.82% | 8.61% (7.92) | fails: +0.7 on clean |

**The expander fails on a real voice.** The `own` set is still unrecorded, so it
was checked on the three real-voice `bench_audio` clips (references unreviewed,
so compared raw vs cleaned, not scored). Raw read "osseous", "pleural effusion"
and "identified"; with the expander the same audio read "osteo", "plural of
fusion" and "ide. Ideify". Synthetic speech is unnaturally level, so 6 dB below
the voice never touches it; a real voice's soft syllables sit there. This is the
slice 2 pre-mortem, confirmed. Every step stays off until a real-voice set
(`own`) shows one helps.

## The bigger noise fix was the model's precision (2026-10-23)

Parakeet had been running its int8 (rounded) copy. At full precision, same
`replay.py` run: `tts` term error 7.13% (was 7.92), `tts_noise` 13.32% (was
24.61), first-word time and lag unchanged. That is a 46% cut in noise, where
the best cleaning step managed 10%. Now the default (`parakeet_precision`), so
the baselines above are int8 numbers: any future step is judged against the
fp32 ones (`data/eval/reports/fp32-<set>.replay.json`).

## Real voices (eka, 2026-10-24): still none passes, all stay off

`eka` is 140 real Indian-English dictations built from Eka Care's medical speech
set (MIT); `eka_quiet` and `eka_noise` are made from it by
`build_degraded_set.py` exactly as the `tts` ones were. Full-precision Parakeet,
`replay.py --limit 30 --condition <step>`. Reports:
`data/eval/reports/eka-<step>-<set>.replay.json`.

| Set | Step | WER | Term error (no step) | Verdict |
|---|---|---|---|---|
| `eka` (clean) | none | 40.46% | 7.78% | |
| `eka_quiet` | none | 54.78% | 15.56% | +7.8 over clean, kept |
| `eka_noise` | none | 54.23% | 17.22% | +9.4 over clean, kept |
| `eka_quiet` | agc | 51.41% | 20.00% (15.56) | fails: term error rises |
| `eka_quiet` | denoise | 59.32% | 23.33% (15.56) | fails: both rise |
| `eka_noise` | denoise | 55.42% | 19.44% (17.22) | fails: both rise |
| `eka` (clean) | denoise | 40.28% | 6.67% (7.78) | passes the clean mark only |
| `eka` (clean) | expander | 40.91% | 12.78% (7.78) | fails: +5.0 on clean |

**The expander garbles real voices, confirmed on 30 speakers' clips.** Same
audio, raw then expander: "chest pain" to "chess pane", "vomiting" to
"womatin'", "popliteal" to "popretial", "nephritic" to "necrotic", and twice a
word came back in Cyrillic ("tuberculosis"). 69 changes in 30 clips: some only case or
punctuation, a few closer to the reference, most worse.

Read the numbers with two cautions. Term error here is a mean of per-clip rates
(replay.py pools errors over all clips from 2026-10-24, as compare.py does, so
later runs are not comparable to these) over few lexicon terms per clip, so one clip flipping moves it about 3 points: a
small difference is noise, but no step moved even in the right direction on a
kept kind, so a bigger sample cannot turn any of these into a pass (the bar is a
quarter cut). And the high WER is partly the references: Eka's labels include
drug brand names and abbreviations ("Telvas AM", "Eurobion plus") that no
general model spells right; the term error, restricted to the radiology
lexicon, is the fair number.
