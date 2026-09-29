"""Deepgram's live (WebSocket) Listen API as an :class:`AsrStream`.

The REST engine in ``deepgram_engine.py`` is handed a finished clip, so the
stream layer has to wait for a chunk to close before anything is kept, and it
re-sends the whole open tail every cycle to build a preview. The live API does
both jobs itself on one socket: audio goes up as it is spoken, interim guesses
come back within a few hundred milliseconds, and settled ("is_final") text
comes back at each endpoint. That is the chunk-once ledger plus
LocalAgreement-2, done server side.

Protocol, as used here (Deepgram "Live Audio" docs):

* connect ``wss://api.deepgram.com/v1/listen?...`` with ``Authorization: Token``;
* send binary frames of raw linear16 PCM;
* receive JSON: ``{"type": "Results", "is_final": bool, "start": s,
  "duration": s, "channel": {"alternatives": [{"transcript", "words"}]}}``;
  finals tile the stream in time;
* send ``{"type": "Finalize"}`` to flush what has been sent (the reply carries
  ``"from_finalize": true``), ``{"type": "KeepAlive"}`` during silence, and
  ``{"type": "CloseStream"}`` to end.

Two threads own the socket: one sends (so :meth:`push` never waits on the
network), one receives (so :meth:`poll` never does). Any failure becomes one
:class:`StreamError` event and the stream goes quiet; the caller decides what
to do about it (``LiveSession`` falls back to decoding chunks).
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from typing import Any, List, Optional, Sequence, Tuple

from src.dictation.asr.types import (
    ProviderUnavailable,
    StreamError,
    StreamEvent,
    StreamFinal,
    StreamInterim,
    Word,
)

logger = logging.getLogger(__name__)

LIVE_URL = "wss://api.deepgram.com/v1/listen"
SAMPLE_RATE = 16000
#: Deepgram closes a socket that has had no audio for about 10 seconds.
_KEEPALIVE_SEC = 5.0
_OPEN_TIMEOUT_SEC = 3.0
#: Milliseconds of silence Deepgram waits before settling an utterance.
#: Short on purpose: settled text is the text that is kept, and a
#: radiologist's between-sentence pause is 300-800ms.
ENDPOINTING_MS = 300

_CLOSE = object()


def live_params(model: str, language: str, keywords: Sequence[str]) -> List[Tuple[str, str]]:
    params = [
        ("model", model),
        ("language", language),
        ("punctuate", "true"),
        ("smart_format", "true"),
        ("encoding", "linear16"),
        ("sample_rate", str(SAMPLE_RATE)),
        ("channels", "1"),
        ("interim_results", "true"),
        ("endpointing", str(ENDPOINTING_MS)),
    ]
    params.extend(("keywords", kw) for kw in keywords)
    return params


class DeepgramStream:
    """One live Deepgram socket. Build with :func:`open_stream`."""

    def __init__(self, ws: Any) -> None:
        self._ws = ws
        self._outbox: "queue.Queue[Any]" = queue.Queue()
        self._events: "queue.Queue[StreamEvent]" = queue.Queue()
        self._lock = threading.Lock()
        self._pushed_sec = 0.0       # audio sent so far, in seconds
        self._settled_sec = 0.0      # end of the last final received
        self._flushed = threading.Event()
        self._dead = threading.Event()
        self._closed = False
        self._sender = threading.Thread(target=self._send_loop, name="deepgram-send", daemon=True)
        self._receiver = threading.Thread(target=self._recv_loop, name="deepgram-recv", daemon=True)
        self._sender.start()
        self._receiver.start()

    # -- AsrStream --------------------------------------------------------

    def push(self, pcm16: bytes) -> None:
        if not pcm16 or self._dead.is_set():
            return
        with self._lock:
            self._pushed_sec += len(pcm16) / (2 * SAMPLE_RATE)
        self._outbox.put(pcm16)

    def poll(self) -> List[StreamEvent]:
        out: List[StreamEvent] = []
        while True:
            try:
                out.append(self._events.get_nowait())
            except queue.Empty:
                return out

    def finalize(self, timeout: float) -> bool:
        if self._dead.is_set():
            return False
        with self._lock:
            target = self._pushed_sec
            if self._settled_sec >= target - 0.02:
                return True
        self._flushed.clear()
        self._outbox.put(json.dumps({"type": "Finalize"}))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self._dead.is_set():
            with self._lock:
                if self._settled_sec >= target - 0.02:
                    return True
            if self._flushed.wait(0.02):
                # The flush reply has arrived: whatever settled is all there is
                # (audio with no speech in it produces no further final).
                return True
        return False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._outbox.put(json.dumps({"type": "CloseStream"}))
        self._outbox.put(_CLOSE)
        self._sender.join(timeout=2.0)
        try:
            self._ws.close()
        except Exception:
            pass
        self._receiver.join(timeout=2.0)

    # -- threads ----------------------------------------------------------

    def _fail(self, message: str) -> None:
        if not self._dead.is_set() and not self._closed:
            logger.warning("Deepgram stream failed: %s", message)
            self._events.put(StreamError(message))
        self._dead.set()

    def _send_loop(self) -> None:
        while True:
            try:
                item = self._outbox.get(timeout=_KEEPALIVE_SEC)
            except queue.Empty:
                # Nothing sent for a while (the page is open, nobody talking):
                # say so, or Deepgram closes the socket after ~10s.
                item = json.dumps({"type": "KeepAlive"})
            if item is _CLOSE:
                return
            if self._dead.is_set():
                continue
            try:
                self._ws.send(item)
            except Exception as exc:
                self._fail(f"send failed: {exc}")

    def _recv_loop(self) -> None:
        try:
            for message in self._ws:
                if isinstance(message, bytes):
                    continue
                self._handle(json.loads(message))
        except Exception as exc:
            self._fail(f"receive failed: {exc}")
            return
        if not self._closed:
            self._fail("server closed the stream")

    def _handle(self, payload: dict) -> None:
        kind = payload.get("type")
        if kind == "Error" or "err_code" in payload:
            self._fail(str(payload.get("description") or payload.get("err_msg") or payload))
            return
        if kind != "Results":
            return
        alt = ((payload.get("channel") or {}).get("alternatives") or [{}])[0]
        text = (alt.get("transcript") or "").strip()
        if not payload.get("is_final"):
            self._events.put(StreamInterim(text))
            return
        start = float(payload.get("start", 0.0))
        end = start + float(payload.get("duration", 0.0))
        words = tuple(
            Word(
                text=w.get("punctuated_word") or w["word"],
                start=float(w["start"]) - start,
                end=float(w["end"]) - start,
                confidence=float(w.get("confidence", 1.0)),
            )
            for w in alt.get("words") or ()
        )
        with self._lock:
            self._settled_sec = max(self._settled_sec, end)
        self._events.put(StreamFinal(text, start, end, words))
        if payload.get("from_finalize"):
            self._flushed.set()


def open_stream(
    api_key: str,
    model: str,
    language: str,
    keywords: Sequence[str] = (),
    url: Optional[str] = None,
) -> DeepgramStream:
    """Connect, or raise. The caller keeps decoding chunks if this raises."""
    from urllib.parse import urlencode

    from websockets.sync.client import connect

    from websockets.exceptions import InvalidStatus

    target = f"{url or LIVE_URL}?{urlencode(live_params(model, language, keywords))}"
    try:
        ws = connect(
            target,
            additional_headers={"Authorization": f"Token {api_key}"},
            open_timeout=_OPEN_TIMEOUT_SEC,
            # Audio is sent continuously; the server's own replies are the
            # liveness signal, and a ping timeout would only duplicate _fail().
            ping_interval=None,
            max_size=None,
        )
    except InvalidStatus as exc:
        if exc.response.status_code in (401, 403):
            raise ProviderUnavailable(
                f"Deepgram rejected the stored API key ({exc.response.status_code})"
            ) from exc
        raise
    # The socket outlives this call (DeepgramStream closes it), so it is
    # entered by hand rather than in a with-block; entering it is what
    # websockets >= 17 asks of a connection used directly.
    return DeepgramStream(ws.__enter__())
