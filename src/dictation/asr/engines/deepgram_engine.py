"""AsrEngine adapter over Deepgram's cloud Listen API, medical model.

The only network-dependent engine on this port: audio leaves the device for
cloud decoding. Wired in as the app's default per explicit product decision
(2026-09-05) — every other engine here is offline; this one stays behind the
same swap-seam so the choice is reversible in one settings/factory edit, not
a rewrite. Always paired with :class:`~.fallback_engine.ChainEngine`, which
falls through to Parakeet/Whisper: a missing key, timeout, or outage degrades
to an offline engine instead of losing the dictation (see factory.py).

Uses Deepgram's pre-recorded REST endpoint (``POST /v1/listen``), not the
streaming websocket API: every engine on this port is already called once per
closed chunk / open tail (``src/dictation/stream/``), synchronously, so this
matches the existing call shape instead of adding a second connection
lifecycle only this engine would own.

``nova-2-medical`` is Deepgram's clinical-dictation model, not general
English — the whole reason to reach for Deepgram over a generic ASR API here.
Never default this to the plain ``nova-2`` model.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, List, Optional, Tuple

import numpy as np

from src.core.keychain import clear_secret, get_secret, store_secret
from src.dictation.asr.types import (
    AsrResult,
    AsrSegment,
    CostModel,
    EngineCaps,
    ProviderUnavailable,
    TranscribeContext,
    Word,
)

logger = logging.getLogger(__name__)

_KEYRING_KEY = "deepgram_api_key"
_LISTEN_URL = "https://api.deepgram.com/v1/listen"
DEFAULT_MODEL = "nova-2-medical"
SAMPLE_RATE = 16000  # matches every other engine on this port

# Deadlines, per call. One flat 30s timeout used to cover everything, so a
# hung socket held the live cycle (and the cycle queued behind it) for half a
# minute before the chain fell back to the local engine: the dictation froze.
# A healthy call costs ~0.2s over a warm connection, so a few seconds is
# already generous; the read budget grows with the clip because the polish
# after Stop sends up to 25s of audio in one call.
_CONNECT_TIMEOUT_SEC = 1.5
_READ_BASE_SEC = 2.0
_READ_PER_AUDIO_SEC = 0.3

# Deepgram's keyword-boosting ("spotlight the decoder onto these words") is a
# Nova-2-family feature: query param ``keywords``, repeated once per term,
# each ``word:intensifier``. Capped at 100 terms per request (Deepgram's own
# limit) and sourced from the app's own curated radiology vocabulary, the
# same spelling authority the fuzzy-correction stage snaps typos to
# (src/resources/radiology_lexicon.txt) — not a hand-picked drug list that
# would drift from it.
_MAX_KEYWORDS = 100
_KEYWORD_INTENSIFIER = 2.5


class DeepgramMissingKeyError(ProviderUnavailable):
    """No Deepgram API key in the OS keychain."""


def store_api_key(api_key: str) -> None:
    store_secret(_KEYRING_KEY, api_key)
    logger.info("Deepgram API key stored in OS keychain")


def get_api_key() -> Optional[str]:
    return get_secret(_KEYRING_KEY)


def clear_api_key() -> None:
    clear_secret(_KEYRING_KEY)


class DeepgramEngine:
    """Cloud ASR via Deepgram's Listen API, tuned for medical dictation."""

    def __init__(self, model_name: str = DEFAULT_MODEL, language: str = "en-US") -> None:
        self.model_name = model_name
        self.language = language

    # -- port -----------------------------------------------------------

    def preload(self) -> None:
        """Build the shared HTTPS client. No model to warm, only the connection.

        Done here, on the main thread at startup, rather than lazily inside the
        first decode: the loop calls :meth:`transcribe` from a worker thread,
        and two threads racing to build the singleton would leave one client
        holding sockets nobody closes.

        Raises :class:`DeepgramMissingKeyError` when no key is stored, so a
        chain learns at startup that this tier will never answer, instead of
        on the first decode of the first dictation.
        """
        _http_client()
        if not get_api_key():
            raise DeepgramMissingKeyError("No Deepgram API key in the OS keychain")

    def capabilities(self) -> EngineCaps:
        # Deepgram reports real per-word confidence; this REST endpoint has
        # no decoder-level vocabulary biasing (Whisper's hotwords).
        # Cost measured against the live API over a kept-alive connection:
        # 0.18s for a 2s clip, 0.16s for a 6s one (see _http_client).
        return EngineCaps(
            word_confidence=True, hotwords=False,
            cost=CostModel(fixed_sec=0.2, per_audio_sec=0.01),
            network=True,
        )

    def usable(self) -> bool:
        """Cheap check, no network: is there a key to call with at all?"""
        return bool(get_api_key())

    def transcribe(self, audio: Any, ctx: TranscribeContext) -> AsrResult:
        pcm = _to_linear16(audio)
        if pcm is None:  # empty waveform - nothing to send
            return AsrResult(text="")

        api_key = get_api_key()
        if not api_key:
            raise DeepgramMissingKeyError(
                "No Deepgram API key in the OS keychain — call "
                "deepgram_engine.store_api_key() first"
            )

        params: List[Tuple[str, Any]] = [
            ("model", self.model_name),
            ("language", self.language),
            ("punctuate", "true"),
            ("smart_format", "true"),
            # Raw linear16 has no container header to read a sample rate from,
            # unlike a WAV file: Deepgram needs encoding/sample_rate/channels
            # as explicit query params here, the Content-Type header alone
            # is not enough and a raw-PCM POST 400s without them (verified
            # against the live API 2026-09-05).
            ("encoding", "linear16"),
            ("sample_rate", str(SAMPLE_RATE)),
            ("channels", "1"),
        ]
        params.extend(("keywords", kw) for kw in _boosted_keywords())

        response = _http_client().post(
            _LISTEN_URL,
            params=params,
            headers={
                "Authorization": f"Token {api_key}",
                "Content-Type": "audio/l16",
            },
            content=pcm,
            timeout=call_timeout(len(pcm) / (2 * SAMPLE_RATE)),
        )
        if response.status_code in (401, 403):
            raise ProviderUnavailable(
                f"Deepgram rejected the stored API key ({response.status_code}); "
                "store a working one with deepgram_engine.store_api_key()"
            )
        response.raise_for_status()
        return _to_result(response.json())


# -- internals ------------------------------------------------------------

def call_timeout(clip_sec: float) -> Any:
    """The deadline for one call carrying *clip_sec* seconds of audio."""
    import httpx

    read = _READ_BASE_SEC + _READ_PER_AUDIO_SEC * max(0.0, clip_sec)
    return httpx.Timeout(read, connect=_CONNECT_TIMEOUT_SEC)


@lru_cache(maxsize=1)
def _http_client() -> Any:
    """One HTTPS connection, kept open for the life of the process.

    The live loop calls this engine once per cycle, so a decode's cost is what
    the radiologist waits on. Measured against the live API on this machine, a
    fresh connection per call costs 1.29s for a 2s clip and 1.36s for a 6s one
    -- almost all of it the TLS handshake, not the transcription. Over a
    kept-alive connection the same two calls cost 0.18s and 0.16s.

    That 1.2s of handshake per decode is what made dictation trail the
    microphone: the preview pacing rule holds the next preview back for as
    long as the last one took, so paying it every cycle stretched updates to
    about five seconds apart, and the ones that ran slow tripped
    ``preview_max_lag_sec`` and switched the preview off outright.
    """
    import httpx  # already a hard dependency (web app / Lightning REST)

    return httpx.Client(
        # Every request passes its own deadline (call_timeout); this is only
        # the ceiling for anything that forgets to.
        timeout=httpx.Timeout(10.0, connect=_CONNECT_TIMEOUT_SEC),
        # A dictation is a burst of calls seconds apart with quiet in between;
        # the expiry has to outlast the quiet or the handshake comes straight
        # back on the first word of the next report.
        limits=httpx.Limits(max_keepalive_connections=4, keepalive_expiry=300.0),
    )


@lru_cache(maxsize=1)
def _boosted_keywords() -> Tuple[str, ...]:
    """``word:intensifier`` entries for Deepgram's ``keywords`` param.

    Read once (the lexicon is static for the life of the process) from the
    same curated radiology wordlist the fuzzy-correction stage already
    trusts — see ``src/dictation/postprocess/CLAUDE.md``. Missing or
    unreadable degrades to no boosting rather than failing the transcribe
    call: the medical model still runs, just without the spotlight.
    """
    try:
        from src.features.file_manager import radiology_lexicon_path

        lines = radiology_lexicon_path().read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.warning("Could not read radiology lexicon for keyword boosting: %s", exc)
        return ()

    terms = [
        line.strip() for line in lines
        if line.strip() and not line.startswith("#")
    ]
    return tuple(f"{term}:{_KEYWORD_INTENSIFIER}" for term in terms[:_MAX_KEYWORDS])


def _to_linear16(audio: Any) -> Optional[bytes]:
    """Normalise the port's audio argument (path / file-like / float32 ndarray)
    into raw 16-bit PCM bytes, mirroring ParakeetEngine's ``_as_input``."""
    if isinstance(audio, str) or hasattr(audio, "read"):
        import soundfile as sf  # already used by worker.py for the same job

        samples, _ = sf.read(audio, dtype="float32")
    else:
        samples = np.asarray(audio, dtype=np.float32)

    if samples.size == 0:
        return None
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def _to_result(payload: dict) -> AsrResult:
    channels = (payload.get("results") or {}).get("channels") or []
    if not channels:
        return AsrResult(text="")
    alternatives = channels[0].get("alternatives") or [{}]
    text = (alternatives[0].get("transcript") or "").strip()
    if not text:
        return AsrResult(text="")

    words = tuple(
        Word(
            text=w["word"],
            start=float(w["start"]),
            end=float(w["end"]),
            confidence=float(w.get("confidence", 1.0)),
        )
        for w in alternatives[0].get("words") or ()
    )
    end = words[-1].end if words else 0.0
    return AsrResult(
        text=text,
        segments=(AsrSegment(text=text, start=0.0, end=end, words=words),),
    )
