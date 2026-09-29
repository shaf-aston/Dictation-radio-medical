"""A local stand-in for Deepgram's live Listen socket, for tests and simulation.

Speaks the same protocol ``src/dictation/asr/engines/deepgram_stream.py``
speaks to the real service (binary linear16 up; JSON ``Results`` with
``is_final`` / ``start`` / ``duration`` / ``from_finalize`` down; ``Finalize``,
``KeepAlive`` and ``CloseStream`` messages), and "recognises" the synthetic
audio of :mod:`scripts.eval.simulate_lag`, whose samples carry the index of
the word they belong to. So it can check the whole live path, socket included,
with no key and no network. It says nothing about recognition quality.

Behaviour modelled on the real service:

* an interim result for the unsettled audio every ``interim_every_sec``;
* a final at each endpoint (``endpointing_ms`` of silence after speech), after
  ``max_utterance_sec`` of unbroken speech, and for ``silence_final_sec`` of
  pure silence (empty text), finals tiling the stream in time;
* ``Finalize`` settles everything received, marked ``from_finalize``;
* a ``latency_sec`` delay in front of every reply;
* ``Token <key>`` must match, or the handshake is refused with 401;
* ``drop_after_sec`` closes the socket once that much audio has arrived.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from scripts.eval.simulate_lag import SR, energy_vad, word_indices


@dataclass
class FakeDeepgramConfig:
    api_key: str = "good-key"
    latency_sec: float = 0.1
    endpointing_ms: int = 300
    interim_every_sec: float = 0.3
    max_utterance_sec: float = 8.0
    silence_final_sec: float = 3.0
    drop_after_sec: Optional[float] = None


class FakeDeepgram:
    """``with FakeDeepgram() as server: ... server.url``"""

    def __init__(self, config: Optional[FakeDeepgramConfig] = None) -> None:
        self.config = config or FakeDeepgramConfig()
        self.connections = 0
        self._server = None
        self._thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        host, port = self._server.socket.getsockname()[:2]
        return f"ws://{host}:{port}/v1/listen"

    def __enter__(self) -> "FakeDeepgram":
        from websockets.sync.server import serve

        self._server = serve(self._handle, "127.0.0.1", 0, process_request=self._check_key)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._server.shutdown()
        self._thread.join(timeout=5)

    # -- server -----------------------------------------------------------

    def _check_key(self, connection, request):
        if request.headers.get("Authorization") != f"Token {self.config.api_key}":
            return connection.respond(401, "Invalid credentials.\n")
        return None

    def _handle(self, ws) -> None:
        self.connections += 1
        cfg = self.config
        audio = np.zeros(0, dtype=np.float32)
        settled = 0            # samples already covered by a final
        last_interim = 0       # sample count at the last interim

        def send(payload: dict) -> None:
            time.sleep(cfg.latency_sec)
            ws.send(json.dumps(payload))

        def result(start: int, end: int, final: bool, from_finalize: bool = False) -> dict:
            clip = audio[start:end]
            ids = word_indices(clip) if end > start else []
            text = " ".join(f"w{i}" for i in ids)
            # Word times are only needed to be plausible and in order.
            span = (end - start) / SR
            words = [
                {"word": f"w{i}", "punctuated_word": f"w{i}", "confidence": 0.95,
                 "start": start / SR + k * span / max(1, len(ids)),
                 "end": start / SR + (k + 1) * span / max(1, len(ids))}
                for k, i in enumerate(ids)
            ]
            return {
                "type": "Results", "is_final": final, "speech_final": final,
                "from_finalize": from_finalize,
                "start": start / SR, "duration": (end - start) / SR,
                "channel": {"alternatives": [{"transcript": text, "confidence": 0.95, "words": words}]},
            }

        for message in ws:
            if isinstance(message, str):
                kind = json.loads(message).get("type")
                if kind in ("Finalize", "CloseStream"):
                    send(result(settled, len(audio), True, from_finalize=True))
                    settled = len(audio)
                if kind == "CloseStream":
                    return
                continue

            block = np.frombuffer(message, dtype="<i2").astype(np.float32) / 32768.0
            audio = np.concatenate([audio, block])
            if cfg.drop_after_sec is not None and len(audio) >= cfg.drop_after_sec * SR:
                ws.close()
                return

            region = audio[settled:]
            marks = energy_vad(region)
            end = len(audio)
            if marks:
                silence_after = len(region) - marks[-1].end_sample
                long_speech = len(region) >= cfg.max_utterance_sec * SR
                # The energy VAD pads each mark by 200ms, like Silero; the
                # endpoint is measured from the real end of speech.
                if silence_after + int(0.2 * SR) >= cfg.endpointing_ms / 1000 * SR or long_speech:
                    send(result(settled, end, True))
                    settled = last_interim = end
                    continue
                if end - last_interim >= cfg.interim_every_sec * SR:
                    send(result(settled, end, False))
                    last_interim = end
            elif len(region) >= cfg.silence_final_sec * SR:
                send(result(settled, end, True))
                settled = last_interim = end


def words_of(text: str) -> List[int]:
    return [int(t[1:]) for t in text.replace(".", " ").split() if t.startswith("w") and t[1:].isdigit()]
