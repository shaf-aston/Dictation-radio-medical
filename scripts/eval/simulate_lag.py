"""Simulate a live dictation's word latency, with no model and no network.

``replay.py`` is the real instrument: real audio, real engine. It needs model
weights or an API key. This one needs neither, so it can judge the part of the
lag that belongs to the LOOP (chunk policy, preview pacing, lanes, streaming)
on any machine, including one that cannot download Whisper:

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
BLOCK_SEC = 0.1
CYCLE_SEC = 0.5

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


def build_dictation(seconds: float, seed: int = 7) -> Tuple[np.ndarray, List[SpokenWord]]:
    """Bursts of 2-10 words, ordinary pauses of 0.35-1.2s, and a longer
    'reading the film' pause of 2.5-3.5s after about one burst in five."""
    rng = random.Random(seed)
    audio: List[np.ndarray] = [np.zeros(int(0.5 * SR), dtype=np.float32)]
    words: List[SpokenWord] = []
    t = 0.5
    while t < seconds:
        for _ in range(rng.randint(2, 10)):
            n = int(WORD_SEC * SR)
            # Each word's samples carry its index; the sign alternates so the
            # "loudness" is constant and any clip boundary is unambiguous.
            value = (len(words) + 1) / 100000.0 + 0.05
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
    ids = np.rint((np.abs(clip[loud]) - 0.05) * 100000.0).astype(int) - 1
    counts = np.bincount(ids)
    need = int(WORD_SEC * SR * 0.5)
    return [i for i, c in enumerate(counts) if c >= need]


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

class SimEngine:
    def __init__(self, cost: CostModel, serial: bool) -> None:
        self.cost = cost
        self._lock = threading.Lock() if serial else None
        self.calls = 0

    def capabilities(self) -> EngineCaps:
        return EngineCaps(word_confidence=True, hotwords=False, cost=self.cost)

    def preload(self) -> None:
        pass

    def transcribe(self, audio, ctx) -> AsrResult:
        clip = np.asarray(audio, dtype=np.float32)
        if self._lock:
            with self._lock:
                return self._decode(clip)
        return self._decode(clip)

    def _decode(self, clip: np.ndarray) -> AsrResult:
        self.calls += 1
        time.sleep(self.cost.estimate(len(clip) / SR))
        ids = word_indices(clip)
        words = tuple(
            Word(f"w{i}", 0.0, 0.0, 0.95) for i in ids
        )
        text = " ".join(w.text for w in words)
        return AsrResult(text=text, segments=(AsrSegment(text, 0.0, len(clip) / SR, words),) if words else ())


# -- the run ----------------------------------------------------------------

def _seen(text: str) -> List[int]:
    out = []
    for token in text.replace(".", " ").replace(",", " ").split():
        token = token.lower()
        if token.startswith("w") and token[1:].isdigit():
            out.append(int(token[1:]))
    return out


def run(engine_name: str, policy: Optional[ChunkPolicy], seconds: float) -> Dict[str, object]:
    audio, words = build_dictation(seconds)
    engine = SimEngine(ENGINES[engine_name], serial=(engine_name == "local"))
    plan = plan_for(engine.capabilities())
    session = LiveSession(
        engine, engine,
        policy=policy or plan.policy,
        preview_min_tail_sec=plan.preview_min_tail_sec,
        cleanup_level="none",
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
        if time.time() - last_cycle >= CYCLE_SEC and not busy.is_set():
            last_cycle = time.time()
            busy.set()
            threading.Thread(target=cycle, daemon=True).start()
    while busy.is_set():
        time.sleep(0.01)

    spoken = {w.index: w.end_sec for w in words}
    kept_lag = [kept[i] - spoken[i] for i in spoken if i in kept]
    shown_lag = [shown[i] - spoken[i] for i in spoken if i in shown]

    def pct(xs: List[float], q: float) -> float:
        xs = sorted(xs)
        return round(xs[min(len(xs) - 1, int(q * len(xs)))], 2) if xs else float("nan")

    used = policy or plan.policy
    return {
        "engine": engine_name,
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
    args = ap.parse_args()
    install_fake_vad()
    policy = None
    if args.policy != "auto":
        lo, soft, force = (float(x) for x in args.policy.split(","))
        policy = ChunkPolicy(min_sec=lo, soft_max_sec=soft, force_cut_sec=force)
    print(json.dumps(run(args.engine, policy, args.seconds)))


if __name__ == "__main__":
    main()
