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
from typing import Any, List, Sequence, Tuple

from src.dictation.asr.types import AsrResult, EngineCaps, ProviderUnavailable, TranscribeContext

logger = logging.getLogger(__name__)


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
            except Exception as exc:
                logger.warning("ASR provider %r failed to preload: %s", name, exc)

    def capabilities(self) -> EngineCaps:
        # What the *primary* provider can offer: a downstream failover to a
        # lesser tier is rare enough that gating every call on the weakest
        # tier's capabilities would cost real accuracy for a rare case.
        return self._providers[0][1].capabilities()

    def transcribe(self, audio: Any, ctx: TranscribeContext) -> AsrResult:
        last_exc: Exception = RuntimeError("ChainEngine has no providers")
        for entry in list(self._providers):
            name, engine = entry
            try:
                return engine.transcribe(audio, ctx)
            except ProviderUnavailable as exc:
                # Permanent until restart: drop it, or every later decode waits
                # on the same rejection first. The last provider is kept so its
                # error still reaches the caller.
                if len(self._providers) > 1:
                    # A fresh list, not remove(): the live and polish threads may share this chain.
                    self._providers = [p for p in self._providers if p is not entry]
                    logger.warning("ASR provider %r unavailable (%s); skipped until restart", name, exc)
                last_exc = exc
            except Exception as exc:
                logger.warning("ASR provider %r failed (%s); trying next", name, exc)
                last_exc = exc
        raise last_exc
