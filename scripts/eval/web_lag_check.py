"""End-to-end word latency through the real web app, with no key or model.

Starts the real FastAPI app (uvicorn), streams synthetic dictation into the
real ``/ws/dictate`` at speaking pace, and times every word from the moment it
was spoken to the moment it was on screen (``shown``) and permanent (``kept``).
Deepgram is replaced by ``fake_deepgram.py`` (live socket) and a 0.2s stand-in
for its REST call; the VAD by the exact detector in ``simulate_lag.py``.
Settings come from a throwaway file, never ``dictation_settings.json``.

Latency and completeness only, never accuracy (see simulate_lag.py).

    python -m scripts.eval.web_lag_check on     # live socket
    python -m scripts.eval.web_lag_check off    # REST chunks, auto chunk plan
    python -m scripts.eval.web_lag_check old    # REST chunks, the old 6/15/20
"""

from __future__ import annotations

import argparse
import json
import pathlib
import tempfile
import threading
import time
from typing import Dict, List

import numpy as np

from scripts.eval import simulate_lag as sim
from scripts.eval.fake_deepgram import FakeDeepgram, FakeDeepgramConfig, words_of

MODES = {
    "on": {"asr_streaming": True},
    "off": {"asr_streaming": False},
    "old": {"asr_streaming": False, "chunk_policy": "manual", "chunk_min_sec": 6.0,
            "chunk_soft_max_sec": 15.0, "chunk_force_cut_sec": 20.0},
}


def _pct(xs: List[float], q: float) -> float:
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 2)


def _install_fakes(mode: str) -> None:
    import src.core.settings as core_settings
    from src.dictation.asr.engines import deepgram_engine
    from src.dictation.stream import live_session, polish
    from src.ui import web_app

    live_session.detect_speech = sim.energy_vad
    polish.detect_speech = sim.energy_vad
    deepgram_engine.get_api_key = lambda: "good-key"
    deepgram_engine._boosted_keywords = lambda: ()
    rest = sim.SimEngine(sim.CostModel(fixed_sec=0.2), serial=False)
    deepgram_engine.DeepgramEngine.transcribe = lambda self, audio, ctx: rest.transcribe(audio, ctx)

    path = pathlib.Path(tempfile.mkdtemp()) / "settings.json"
    path.write_text(json.dumps(MODES[mode]))
    core_settings.settings_file = lambda: path
    web_app.settings_file = lambda: path
    web_app._warm_up_singletons = lambda: None


def run(mode: str, seconds: float, latency: float, port: int) -> Dict[str, object]:
    import uvicorn
    from websockets.sync.client import connect

    from src.dictation.asr.engines import deepgram_stream
    from src.ui import web_app

    _install_fakes(mode)
    audio, spoken = sim.build_dictation(seconds, seed=21)
    shown: Dict[int, float] = {}
    kept: Dict[int, float] = {}
    with FakeDeepgram(FakeDeepgramConfig(latency_sec=latency)) as server:
        deepgram_stream.LIVE_URL = server.url
        app_server = uvicorn.Server(uvicorn.Config(web_app.app, host="127.0.0.1", port=port, log_level="warning"))
        threading.Thread(target=app_server.run, daemon=True).start()
        while not app_server.started:
            time.sleep(0.05)
        with connect(f"ws://127.0.0.1:{port}/ws/dictate", max_size=None) as ws:
            started = time.time()

            def speak() -> None:
                block = sim.SR // 10
                for i in range(0, len(audio), block):
                    if (delay := started + i / sim.SR - time.time()) > 0:
                        time.sleep(delay)
                    ws.send((np.clip(audio[i : i + block], -1, 1) * 32767).astype("<i2").tobytes())
                ws.send(json.dumps({"command": "stop"}))

            threading.Thread(target=speak, daemon=True).start()
            while True:
                msg = json.loads(ws.recv())
                now = time.time() - started
                if msg["type"] == "partial":
                    for i in words_of(msg["committed"].lower()):
                        kept.setdefault(i, now)
                        shown.setdefault(i, now)
                    for i in words_of(msg["preview"].lower()):
                        shown.setdefault(i, now)
                elif msg["type"] == "stopped":
                    stopped_at, handback = now, words_of(msg["text"].lower())
                elif msg["type"] == "final":
                    final_at, final = now, words_of(msg["text"].lower())
                    break
        app_server.should_exit = True

    end = {w.index: w.end_sec for w in spoken}
    kept_lag = [kept[i] - end[i] for i in end if i in kept]
    shown_lag = [shown[i] - end[i] for i in end if i in shown]
    everything = [w.index for w in spoken]
    return {
        "mode": mode, "words": len(spoken), "kept_before_stop": len(kept_lag),
        "kept_p50": _pct(kept_lag, 0.5), "kept_p95": _pct(kept_lag, 0.95),
        "shown_p50": _pct(shown_lag, 0.5), "shown_p95": _pct(shown_lag, 0.95),
        "first_word": round(min(shown.values()) - spoken[0].end_sec, 2),
        "handback_after_audio_end": round(stopped_at - len(audio) / sim.SR, 2),
        "handback_complete": handback == everything,
        "final_complete": final == everything,
        "final_after_handback": round(final_at - stopped_at, 2),
        "live_sockets": server.connections,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=sorted(MODES))
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--latency", type=float, default=0.15, help="fake Deepgram reply delay")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    print(json.dumps(run(args.mode, args.seconds, args.latency, args.port)))


if __name__ == "__main__":
    main()
