"""ChainEngine: an ordered list of AsrEngine providers, one port, N tiers.

The pattern this exists for: any provider can go down (bad key, network,
rate limit, an optional package not installed), and dictation must never
stop because of it. Each entry is tried in the order given; a failure moves
to the next entry, never mid-chunk quality grading, the app has no ground
truth to grade a decode against, only whether the call itself succeeded.
Mirrors ``llm_cleanup.py``'s "degrade, don't crash" rule stretched across
more than two tiers, so adding provider #4 later is one line in
``factory.py``, not a new wrapper class.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Sequence, Tuple

from src.dictation.asr.types import AsrResult, EngineCaps, ProviderUnavailable, TranscribeContext

logger = logging.getLogger(__name__)

# Providers rejected with ProviderUnavailable this process (dead API key, auth
# failure): permanent until restart, and shared across every ChainEngine
# instance. web_app builds one ChainEngine per model name (live + polish), so
# without this each instance paid its own 401 round trip on every decode --
# the per-instance drop below only ever helped the instance that had already
# eaten the cost once. Keyed by provider name, not by instance.
_DEAD_PROVIDERS: set = set()

# A circuit breaker for the other kind of failure: a timeout, a dropped
# connection, a 5xx. Those may clear on their own, so they are not permanent,
# but paying one on EVERY decode during an outage put a network deadline in
# front of each live update and froze the dictation. After TRIP_AFTER failures
# in a row a provider is skipped for TRIP_SEC; the first call after that
# window is a probe, and one more failure re-opens the breaker straight away.
# Shared process-wide and keyed by name, like _DEAD_PROVIDERS.
TRIP_AFTER = 2
TRIP_SEC = 30.0
_FAILURES: Dict[str, int] = {}
_TRIPPED_UNTIL: Dict[str, float] = {}
_clock = time.monotonic


def _tripped(name: str) -> bool:
    return _TRIPPED_UNTIL.get(name, 0.0) > _clock()


def _note_failure(name: str) -> None:
    _FAILURES[name] = _FAILURES.get(name, 0) + 1
    if _FAILURES[name] >= TRIP_AFTER:
        _TRIPPED_UNTIL[name] = _clock() + TRIP_SEC
        logger.warning(
            "ASR provider %r failed %d times in a row; skipped for %.0fs",
            name, _FAILURES[name], TRIP_SEC,
        )


def _note_success(name: str) -> None:
    _FAILURES.pop(name, None)
    _TRIPPED_UNTIL.pop(name, None)


def reset_provider_health() -> None:
    """Forget every dead and tripped provider (tests, and a key re-entered)."""
    _DEAD_PROVIDERS.clear()
    _FAILURES.clear()
    _TRIPPED_UNTIL.clear()


def provider_usable(name: str) -> bool:
    """Neither rejected for good nor inside a breaker window."""
    return name not in _DEAD_PROVIDERS and not _tripped(name)


class ChainEngine:
    """Satisfies the :class:`AsrEngine` port by trying providers in order."""

    def __init__(self, providers: Sequence[Tuple[str, Any]]) -> None:
        if not providers:
            raise ValueError("ChainEngine needs at least one provider")
        self._providers: List[Tuple[str, Any]] = list(providers)

    def preload(self) -> None:
        # Every tier is warmed, not just the primary: a failover mid-dictation
        # must not pay a cold-load cost the radiologist feels as a stall.
        for name, engine in self._providers:
            try:
                engine.preload()
            except ProviderUnavailable as exc:
                # Known dead before the first word (no API key): say so now,
                # so capabilities() already describes the engine that will
                # really answer, and no decode pays to find out.
                _DEAD_PROVIDERS.add(name)
                logger.warning("ASR provider %r unavailable (%s); skipped until restart", name, exc)
            except Exception as exc:
                logger.warning("ASR provider %r failed to preload: %s", name, exc)

    def capabilities(self) -> EngineCaps:
        # What the provider that will actually answer the next call can offer:
        # the first one still usable. With no Deepgram key that is the local
        # engine, and the stream layer must size its chunks for THAT engine's
        # price, not for a cloud engine that will never be called.
        return self.active()[1].capabilities()

    def active(self) -> Tuple[str, Any]:
        """``(name, engine)`` of the provider the next call will try first.

        A provider may offer ``usable()``, a cheap local check (Deepgram: is a
        key stored at all?). A no there is as final as a 401, so it is
        remembered the same way.
        """
        for name, engine in self._providers:
            if not provider_usable(name):
                continue
            usable = getattr(engine, "usable", None)
            if usable is not None and not usable():
                _DEAD_PROVIDERS.add(name)
                continue
            return name, engine
        return self._providers[-1]

    def transcribe(self, audio: Any, ctx: TranscribeContext) -> AsrResult:
        last_exc: Exception = RuntimeError("ChainEngine has no providers")
        entries = list(self._providers)
        for i, entry in enumerate(entries):
            name, engine = entry
            # Skip a provider already known dead process-wide -- unless it is
            # the only one left, so its error still reaches the caller.
            # The same for one inside a breaker window.
            if not provider_usable(name) and i < len(entries) - 1:
                continue
            try:
                result = engine.transcribe(audio, ctx)
            except ProviderUnavailable as exc:
                # Permanent until restart: drop it, or every later decode waits
                # on the same rejection first. The last provider is kept so its
                # error still reaches the caller.
                _DEAD_PROVIDERS.add(name)
                if len(self._providers) > 1:
                    # A fresh list, not remove(): the live and polish threads may share this chain.
                    self._providers = [p for p in self._providers if p is not entry]
                    logger.warning("ASR provider %r unavailable (%s); skipped until restart", name, exc)
                last_exc = exc
            except Exception as exc:
                logger.warning("ASR provider %r failed (%s); trying next", name, exc)
                _note_failure(name)
                last_exc = exc
            else:
                _note_success(name)
                return result
        raise last_exc
