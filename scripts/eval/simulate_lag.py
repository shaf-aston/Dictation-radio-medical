"""Simulate a live dictation's word latency, with no model and no network.

``replay.py`` is the real instrument: real audio, real engine. It needs model
weights or an API key. This one needs neither, so it can judge the part of the
lag that belongs to the LOOP (chunk policy, preview pacing, lanes, streaming)
on any machine, including one that cannot download Whisper (it is how the
"two lanes" preview thread was measured and refuted: see
docs/dictation-speed-review.md, 2026-09-27):

* the audio is synthetic: bursts of "speech" separated by pauses, each word a
  run of samples carrying its own index, so a decode can say exactly which
  words it heard;
* the VAD is an exact energy detector over that signal (padded like Silero);
* the engine sleeps for what its :class:`CostModel` says a call costs, then
  returns the words whose samples are mostly inside the clip. A local engine
  holds a lock while it "decodes" (ctranslate2 ``num_workers=1`` serialises
  calls); a cloud engine does not.

It drives the real ``LiveSession`` at true speaking pace, on the same
schedule the WebSocket uses, and reports per-word lag: from the moment a word
finished being spoken to the moment it was on screen (``shown``) and to the
moment it became permanent (``kept``). It says NOTHING about accuracy.

Run:
    python -m scripts.eval.simulate_lag --engine cloud --policy auto
    python -m scripts.eval.simulate_lag --engine local --policy 6,15,20
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.dictation.asr.types import AsrResult, AsrSegment, CostModel, EngineCaps, Word
from src.dictation.stream import live_session as live_session_mod
from src.dictation.stream import polish as polish_mod
from src.dictation.stream.live_session import LiveSession
from src.dictation.stream.policy import plan_for
from src.dictation.stream.segmenter import ChunkPolicy
from src.dictation.stream.vad import SpeechMark

SR = 16000
WORD_SEC = 0.35
BLOCK_SEC = 0.05
CYCLE_SEC = 0.5
SEED = 7
_AMP_BASE = 0.05
_AMP_STEP = 0.0005

ENGINES = {
    # Deepgram over a warm connection (deepgram_engine._http_client numbers).
    "cloud": CostModel(fixed_sec=0.2, per_audio_sec=0.01),
    # tiny.en on the development machine (dictation-speed-review.md).
    "local": CostModel(fixed_sec=0.8, per_audio_sec=0.04),
}


# -- the synthetic dictation ------------------------------------------------

@dataclass(frozen=True)
class SpokenWord:
    index: int
    start_sec: float
    end_sec: float


def build_dictation(seconds: float, seed: int = SEED) -> Tuple[np.ndarray, List[SpokenWord]]:
    """Bursts of 2-10 words, ordinary pauses of 0.35-1.2s, and a longer
    'reading the film' pause of 2.5-3.5s after about one burst in five."""
    rng = random.Random(seed)
    audio: List[np.ndarray] = [np.zeros(int(0.5 * SR), dtype=np.float32)]
    words: List[SpokenWord] = []
    t = 0.5
    while t < seconds:
        for _ in range(rng.randint(2, 10)):
            n = int(WORD_SEC * SR)
            # Each word's samples carry its index, coarsely enough (16 LSB of
            # 16-bit PCM per step) to survive being written to a WAV; the sign
            # alternates so the "loudness" is constant.
            value = _AMP_BASE + len(words) * _AMP_STEP
            samples = np.full(n, value, dtype=np.float32)
            samples[1::2] *= -1
            audio.append(samples)
            words.append(SpokenWord(len(words), t, t + WORD_SEC))
            t += WORD_SEC
        pause = rng.uniform(2.5, 3.5) if rng.random() < 0.2 else rng.uniform(0.35, 1.2)
        audio.append(np.zeros(int(pause * SR), dtype=np.float32))
        t += pause
    return np.concatenate(audio), words


def word_indices(clip: np.ndarray) -> List[int]:
    """Which words a clip mostly contains, in order."""
    loud = np.abs(clip) > 0.01
    if not loud.any():
        return []
    ids = np.rint((np.abs(clip[loud]) - _AMP_BASE) / _AMP_STEP).astype(int)
    counts = np.bincount(ids)
    need = int(WORD_SEC * SR * 0.5)
    return [i for i, c in enumerate(counts) if c >= need]


def word_spans(clip: np.ndarray) -> Dict[int, Tuple[float, float]]:
    """Where each word sits in the clip, in seconds: real word timings."""
    loud = np.flatnonzero(np.abs(clip) > 0.01)
    if not len(loud):
        return {}
    ids = np.rint((np.abs(clip[loud]) - _AMP_BASE) / _AMP_STEP).astype(int)
    spans: Dict[int, Tuple[float, float]] = {}
    for i in np.unique(ids):
        where = loud[ids == i]
        spans[int(i)] = (where[0] / SR, (where[-1] + 1) / SR)
    return spans


def energy_vad(audio: np.ndarray, min_silence_ms: int = 300, speech_pad_ms: int = 200) -> List[SpeechMark]:
    """Exact VAD for the synthetic signal, padded and merged like Silero."""
    frame = int(0.03 * SR)
    loud = np.abs(audio) > 0.01
    n = len(audio) // frame
    if n == 0:
        return []
    voiced = loud[: n * frame].reshape(n, frame).any(axis=1)
    marks: List[List[int]] = []
    for i, v in enumerate(voiced):
        if not v:
            continue
        s, e = i * frame, (i + 1) * frame
        if marks and s - marks[-1][1] < int(min_silence_ms / 1000 * SR):
            marks[-1][1] = e
        else:
            marks.append([s, e])
    pad = int(speech_pad_ms / 1000 * SR)
    return [SpeechMark(max(0, s - pad), min(len(audio), e + pad)) for s, e in marks]


# -- the engine -------------------------------------------------------------

#: Two decodes overlapping on one Whisper model with num_workers=2: measured
#: 1.775s for two 10s decodes that take 2.132s back to back, i.e. each call
#: runs ~1.67x slower while the other is running (lag-map-2026-09-02.md, #7).
OVERLAP_SLOWDOWN = 1.775 / (2.132 / 2)


class SimEngine:
    """*serial*: calls queue (ctranslate2 num_workers=1). *workers* > 1: up to
    that many run at once, each slowed by OVERLAP_SLOWDOWN while overlapped.
    A cloud engine is neither: calls are independent network requests."""

    #: Local engines running right now, across every instance: two local models
    #: (tiny.en live, small.en polishing in the background) share the CPU.
    _cpu_busy = 0
    _cpu_lock = threading.Lock()

    def __init__(
        self, cost: CostModel, serial: bool, workers: int = 1,
        name: str = "sim", weak_every: int = 0,
    ) -> None:
        self.cost = cost
        self.name = name
        # Every third group of *weak_every* words is heard badly (0.55), so a
        # share of chunks falls under the polish ceiling as real audio does.
        self.weak_every = weak_every
        self.workers = workers
        slots = 1 if serial and workers <= 1 else workers if serial else 0
        self._lock = threading.BoundedSemaphore(slots) if slots else None
        self._busy = 0
        self._busy_lock = threading.Lock()
        self.calls = 0

    def capabilities(self) -> EngineCaps:
        return EngineCaps(word_confidence=True, hotwords=False, cost=self.cost)

    def identity(self) -> str:
        return self.name

    def preload(self) -> None:
        pass

    def transcribe(self, audio, ctx) -> AsrResult:
        clip = np.asarray(audio, dtype=np.float32)
        if self._lock:
            with self._lock:
                return self._decode(clip)
        return self._decode(clip)

    def _decode(self, clip: np.ndarray) -> AsrResult:
        local = self._lock is not None
        with self._busy_lock:
            self.calls += 1
            self._busy += 1
        with SimEngine._cpu_lock:
            if local:
                SimEngine._cpu_busy += 1
            overlapped = local and SimEngine._cpu_busy > 1
        try:
            cost = self.cost.estimate(len(clip) / SR)
            time.sleep(cost * (OVERLAP_SLOWDOWN if overlapped else 1.0))
        finally:
            with self._busy_lock:
                self._busy -= 1
            with SimEngine._cpu_lock:
                if local:
                    SimEngine._cpu_busy -= 1
        ids = word_indices(clip)
        spans = word_spans(clip)
        words = tuple(
            Word(f"w{i}", spans[i][0], spans[i][1], self._confidence(i)) for i in ids
        )
        text = " ".join(w.text for w in words)
        return AsrResult(text=text, segments=(AsrSegment(text, 0.0, len(clip) / SR, words),) if words else ())


# -- the run ----------------------------------------------------------------

    def _confidence(self, index: int) -> float:
        if self.weak_every and (index // self.weak_every) % 3 == 0:
            return 0.55
        return 0.95


def _seen(text: str) -> List[int]:
    out = []
    for token in text.replace(".", " ").replace(",", " ").split():
        token = token.lower()
        if token.startswith("w") and token[1:].isdigit():
            out.append(int(token[1:]))
    return out


def run(
    engine_name: str, policy: Optional[ChunkPolicy], seconds: float,
    workers: int = 1, cycle_sec: float = CYCLE_SEC,
    seed: int = SEED, final: str = "same", background: bool = True,
) -> Dict[str, object]:
    audio, words = build_dictation(seconds, seed)
    local = engine_name == "local"
    engine = SimEngine(ENGINES[engine_name], serial=local, workers=workers, name="live", weak_every=4)
    # "small": a distinct accurate model (small.en's measured ~3.8s per call)
    # re-decodes weak chunks; "same": the live engine answers both, as with a
    # Deepgram key, and the polish is skipped.
    final_engine = (
        SimEngine(CostModel(fixed_sec=3.8, per_audio_sec=0.04), serial=True, name="small")
        if final == "small" else engine
    )
    plan = plan_for(engine.capabilities())
    session = LiveSession(
        engine, final_engine,
        policy=policy or plan.policy,
        preview_min_tail_sec=plan.preview_min_tail_sec,
        cleanup_level="none",
        background_polish=background,
    )
    shown: Dict[int, float] = {}
    kept: Dict[int, float] = {}
    started = time.time()
    lock = threading.Lock()
    busy = threading.Event()

    def cycle() -> None:
        try:
            update = session.cycle()
            now = time.time() - started
            if update is not None:
                with lock:
                    for i in _seen(update.committed):
                        kept.setdefault(i, now)
                        shown.setdefault(i, now)
                    for i in _seen(update.preview):
                        shown.setdefault(i, now)
        finally:
            busy.clear()

    block = int(BLOCK_SEC * SR)
    last_cycle = 0.0
    for offset in range(0, len(audio), block):
        target = started + offset / SR
        if (delay := target - time.time()) > 0:
            time.sleep(delay)
        session.feed(audio[offset : offset + block])
        if time.time() - last_cycle >= cycle_sec and not busy.is_set():
            last_cycle = time.time()
            busy.set()
            threading.Thread(target=cycle, daemon=True).start()
    while busy.is_set():
        time.sleep(0.01)
    audio_end = time.time() - started
    stop = time.time()
    session.close_open_tail_fast()
    handback_sec = time.time() - stop
    final_text = session.finalize()
    settled_sec = time.time() - stop

    spoken = {w.index: w.end_sec for w in words}
    kept_lag = [kept[i] - spoken[i] for i in spoken if i in kept]
    shown_lag = [shown[i] - spoken[i] for i in spoken if i in shown]

    def pct(xs: List[float], q: float) -> float:
        xs = sorted(xs)
        return round(xs[min(len(xs) - 1, int(q * len(xs)))], 2) if xs else float("nan")

    used = policy or plan.policy
    return {
        "final": final,
        "background": background,
        "handback_sec": round(handback_sec, 2),
        "settled_after_stop_sec": round(settled_sec, 2),
        "final_complete": _seen(final_text) == [w.index for w in words],
        "audio_sec": round(audio_end, 1),
        "engine": engine_name,
        "workers": workers,
        "cycle_sec": cycle_sec,
        "policy": f"{used.min_sec:g}/{used.soft_max_sec:g}/{used.force_cut_sec:g}",
        "words": len(words),
        "kept_before_stop": len(kept_lag),
        "kept_p50": pct(kept_lag, 0.5),
        "kept_p95": pct(kept_lag, 0.95),
        "shown_p50": pct(shown_lag, 0.5),
        "shown_p95": pct(shown_lag, 0.95),
        "first_word_sec": round(min(shown.values()) - words[0].end_sec, 2) if shown else None,
        "decode_calls": engine.calls,
        "mean_kept_lag": round(statistics.fmean(kept_lag), 2) if kept_lag else None,
    }


def install_fake_vad() -> None:
    live_session_mod.detect_speech = energy_vad
    polish_mod.detect_speech = energy_vad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=sorted(ENGINES), default="cloud")
    ap.add_argument("--policy", default="auto", help='"auto" or min,soft,force seconds')
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--workers", type=int, default=1, help="concurrent decodes the local engine allows")
    ap.add_argument("--seed", type=int, default=SEED, help="which synthetic dictation")
    ap.add_argument("--final", choices=("same", "small"), default="same",
                    help="accurate engine: the live one, or a distinct small.en-priced one")
    ap.add_argument("--no-background", action="store_true", help="polish only after Stop")
    ap.add_argument("--cycle", type=float, default=CYCLE_SEC, help="seconds between cycles (live_cycle_sec)")
    args = ap.parse_args()
    install_fake_vad()
    policy = None
    if args.policy != "auto":
        lo, soft, force = (float(x) for x in args.policy.split(","))
        policy = ChunkPolicy(min_sec=lo, soft_max_sec=soft, force_cut_sec=force)
    print(json.dumps(run(args.engine, policy, args.seconds, args.workers, args.cycle, args.seed,
                         args.final, not args.no_background)))


if __name__ == "__main__":
    main()
