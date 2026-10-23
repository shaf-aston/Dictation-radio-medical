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
- Latency: words appear no more than 0.05 s later (the design costs 0.016 s).
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

# factory.py
def build_chain(cfg: dict, sr: int) -> Chain
    """cfg is settings["audio_conditioning"]; each step's section is merged over
    get_default's, so an old settings file still gets every knob. Unknown step
    name, or a zero/negative value where one must be positive -> ValueError."""
```

Fixed order, set in `factory.py`, not a setting: **highpass -> denoise -> expander -> agc**.
Gating before gain, so the volume step never turns up the gaps between words.

| Step | What it does | Knobs |
|---|---|---|
| highpass | gain 0 below `cutoff_hz`, smooth ramp up to it; stateless | `cutoff_hz` 80 |
| denoise | tracks each bin's noise floor (falls at once, rises `noise_rise_db_per_sec`); spectral subtraction, never below `min_gain_db` (the floor is the "blend some original back" fix from the research) | `strength` 1.0, `min_gain_db` -12, `noise_rise_db_per_sec` 3 |
| expander | follows the main voice's level (`PeakFollower`); frames more than `below_voice_db` under it are turned down, at most `max_cut_db` | `below_voice_db` 20, `max_cut_db` 15, `voice_release_db_per_sec` 6 |
| agc | moves the voice toward `target_dbfs`, at most `slew_db_per_sec`, never above `max_gain_db`; adapts only on frames within 10 dB of the voice level | `target_dbfs` -20, `max_gain_db` 24, `slew_db_per_sec` 6 |

Settings (`src/core/settings.py` `_DEFAULTS`):

```python
"audio_conditioning": {
    "window_samples": 512,
    "highpass": {"on": False, "cutoff_hz": 80.0},
    "denoise":  {"on": False, "strength": 1.0, "min_gain_db": -12.0, "noise_rise_db_per_sec": 3.0},
    "expander": {"on": False, "below_voice_db": 20.0, "max_cut_db": 15.0, "voice_release_db_per_sec": 6.0},
    "agc":      {"on": False, "target_dbfs": -20.0, "max_gain_db": 24.0, "slew_db_per_sec": 6.0},
},
```

**LiveSession** gets `conditioner: Optional[Chain] = None`. `feed` appends
`conditioner.process(block)` instead of `block`; `close_open_tail_fast` first
appends `conditioner.flush()` (every caller runs it before `finalize`). The append
itself stays one private method, so the raw and cleaned paths cannot differ.
One Chain per session, built at session construction: it learns the room, and a
shared one would carry one dictation's noise into the next.

Callers: `worker.py:247` and `web_app.py:1038` pass `build_chain(settings.get(...), SAMPLE_RATE)`;
`replay.py` passes one when `--condition` names steps; `simulate_lag.py` passes nothing.
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
