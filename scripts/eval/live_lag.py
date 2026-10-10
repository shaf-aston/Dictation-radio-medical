"""How long each word takes to reach the screen, on THIS machine, end to end.

``replay.py`` times when words become permanent (commit lag). The number a
radiologist describes as "slow" is earlier than that: how long after a word is
spoken it is first on screen, preview included. This measures both, plus how
often a word already on screen changes (the price of a faster preview).

It builds the session with the web app's own ``_live_session`` (same engines,
same chunk plan, same preview mode as a real dictation), feeds the recording in
20ms frames at true speaking pace, and runs ``cycle()`` on a worker thread every
``live_cycle_sec`` exactly as ``/ws/dictate`` does.

When was each word spoken? One whole-file decode with the same engine gives
each word's start; a word ends at the next word's start or where its stretch
of speech ends (VAD), whichever is first, so a word before a pause is not timed
as ending after the pause. A word counts as shown once the screen text, aligned
against that sequence, contains it. Words the live loop never gets right are
left out, so read ``matched`` next to the percentiles.

Optional ``--load N`` runs N busy processes alongside, which is how the
machine this was measured on behaves with a browser, a PACS viewer and a
virus scanner open.

Run:
    python -m scripts.eval.live_lag --audio data/bench_audio/chest_long.wav
    python -m scripts.eval.live_lag --audio my_dictation.wav --load 2
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
import sys
import threading
import time
from typing import List, Optional, Tuple

import numpy as np

from scripts.eval.replay import load_audio
from src.core.settings import Settings
from src.dictation.asr import TranscribeContext
from src.dictation.stream.vad import SAMPLE_RATE, detect_speech

FRAME_SEC = 0.02


def _norm(word: str) -> str:
    return re.sub(r"[^\w]", "", word.lower())


def spoken_words(engine, audio: np.ndarray) -> Tuple[List[str], List[float]]:
    """Each word of a whole-file decode, and roughly when it finished."""
    result = engine.transcribe(audio, TranscribeContext())
    words = [w for seg in result.segments for w in seg.words]
    if not words:
        raise SystemExit("The engine heard no words in this recording")
    marks = [
        (m.start_sample / SAMPLE_RATE, m.end_sample / SAMPLE_RATE)
        for m in detect_speech(audio, min_silence_ms=150, speech_pad_ms=0)
    ]

    def speech_end(t: float) -> float:
        for start, end in marks:
            if start - 0.1 <= t <= end + 0.1:
                return end
        return t + 0.4

    starts = [w.start for w in words]
    ends = [
        min(starts[i + 1] if i + 1 < len(starts) else float("inf"), speech_end(starts[i]))
        for i in range(len(words))
    ]
    return [_norm(w.text) for w in words], ends


def first_seen(reference: List[str], text: str, seen: List[Optional[float]], at: float) -> None:
    hyp = [_norm(w) for w in text.split()]
    matcher = difflib.SequenceMatcher(None, reference, hyp, autojunk=False)
    for block in matcher.get_matching_blocks():
        for i in range(block.a, block.a + block.size):
            if seen[i] is None:
                seen[i] = at


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(p * len(ordered)))], 2)


def run(audio: np.ndarray, settings: Settings) -> dict:
    from src.ui import web_app  # noqa: PLC0415 - the app's own session wiring

    session = web_app._live_session(settings, web_app._current_preferences())
    session.live_engine.preload()
    reference, ends = spoken_words(session.live_engine, audio)
    cycle_sec = float(settings.get("live_cycle_sec"))

    updates: List[Tuple[float, str, str]] = []
    frame = int(FRAME_SEC * SAMPLE_RATE)
    worker: Optional[threading.Thread] = None
    last_cycle = 0.0
    fed = 0

    def cycle() -> None:
        update = session.cycle()
        if update is not None:
            updates.append((time.perf_counter() - t0, update.committed, update.preview))

    cpu0 = time.process_time()
    t0 = time.perf_counter()
    while fed < len(audio):
        if int((time.perf_counter() - t0) * SAMPLE_RATE) < fed + frame:
            time.sleep(0.002)
            continue
        session.feed(audio[fed : fed + frame])
        fed += frame
        now = time.perf_counter() - t0
        if now - last_cycle >= cycle_sec and (worker is None or not worker.is_alive()):
            last_cycle = now
            worker = threading.Thread(target=cycle)
            worker.start()
    stopped = time.perf_counter() - t0
    if worker is not None:
        worker.join()
    session.close_open_tail_fast()
    handed = time.perf_counter() - t0
    updates.append((handed, session.committed_text(), ""))
    session.finalize()
    cpu = time.process_time() - cpu0

    shown: List[Optional[float]] = [None] * len(reference)
    kept: List[Optional[float]] = [None] * len(reference)
    changed = 0
    previous_committed, previous_preview = None, []
    for at, committed, preview in updates:
        first_seen(reference, f"{committed} {preview}", shown, at)
        first_seen(reference, committed, kept, at)
        words = [_norm(w) for w in preview.split()]
        if committed == previous_committed:
            changed += sum(
                1 for i, w in enumerate(previous_preview) if i >= len(words) or words[i] != w
            )
        previous_committed, previous_preview = committed, words

    shown_lag = [shown[i] - ends[i] for i in range(len(ends)) if shown[i] is not None]
    kept_lag = [kept[i] - ends[i] for i in range(len(ends)) if kept[i] is not None]
    audio_min = len(audio) / SAMPLE_RATE / 60
    return {
        "audio_sec": round(len(audio) / SAMPLE_RATE, 1),
        "words": len(reference),
        "matched": len(shown_lag),
        "first_words_sec": next(
            (round(at, 2) for at, c, p in updates if c or p), None
        ),
        "shown_p50": percentile(shown_lag, 0.5),
        "shown_p90": percentile(shown_lag, 0.9),
        "kept_p50": percentile(kept_lag, 0.5),
        "kept_p90": percentile(kept_lag, 0.9),
        "handed_back_after_stop_sec": round(handed - stopped, 2),
        "shown_words_changed_per_min": round(changed / audio_min, 1),
        "cpu_sec_per_audio_sec": round(cpu / (len(audio) / SAMPLE_RATE), 2),
        "preview_mode": settings.get("preview_mode"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--audio", required=True, help="a recording to dictate (any rate)")
    ap.add_argument("--load", type=int, default=0, help="busy processes to run alongside")
    ap.add_argument("--preview-mode", choices=("instant", "confirmed"),
                    help="override the preview_mode setting for this run only")
    args = ap.parse_args()

    settings = Settings()
    if args.preview_mode:
        # In memory only: an A/B run must not rewrite the user's settings file.
        settings._data["preview_mode"] = args.preview_mode
    burners = [
        subprocess.Popen([sys.executable, "-c", "while 1: pass"]) for _ in range(args.load)
    ]
    try:
        print(json.dumps(run(load_audio(args.audio), settings), indent=2))
    finally:
        for burner in burners:
            burner.kill()


if __name__ == "__main__":
    main()
