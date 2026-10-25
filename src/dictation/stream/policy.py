"""How the live loop is shaped, chosen from what the engine says a call costs.

Every constant here used to be one hard-coded answer, sized for local Whisper,
whose every call pays ~1-4s however little audio it is handed (the 30s pad).
Few long chunks amortise that. But the default engine became Deepgram at
~0.2s a call, and the same long chunks then only made the kept text trail the
microphone by the chunk length: the decode was fast and the word still waited
6 to 15 seconds for its chunk to close. So the engine declares its price
(:class:`~src.dictation.asr.types.CostModel`) and this module turns the price
into a shape. Pure: no settings file, no engine calls, no I/O.

Both rows are measured starting points, not settled truths: any change is
judged by ``scripts/eval/replay.py`` against the medical-term error veto.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from src.dictation.asr.types import EngineCaps
from src.dictation.stream.segmenter import ChunkPolicy

#: A call cheaper than this is "cheap": short chunks cost almost nothing more
#: in decode, so the loop commits at the first real pause instead of waiting.
CHEAP_CALL_SEC = 0.5

#: Local Whisper-class engines. 2s / 5s is the policy the replay harness
#: measured best on tiny.en live + small.en polish (docs/dictation-accuracy.md,
#: 2026-09-06): commit lag p50 4.78s -> 2.93s AND lower WER / term error than
#: the old 6 / 15. The 20s force cut is unchanged: it only fires on an unbroken
#: monologue, and cutting mid-speech any sooner risks cutting a word.
LOCAL_POLICY = (2.0, 5.0, 20.0)

#: Cheap (cloud) engines. Commit at the first pause after a second of speech.
#: Not yet measured against the accuracy veto on real Deepgram audio: this
#: container has no key, so it is the PRD's starting row, to be confirmed with
#: ``replay.py --engine deepgram`` before being trusted further.
CHEAP_POLICY = (1.0, 3.0, 8.0)

#: The shortest open tail worth a preview decode. For an engine whose call is
#: priced flat, decoding a fraction of a second spends a whole call to show
#: almost nothing; for a cheap engine that call is nearly free.
LOCAL_PREVIEW_MIN_TAIL_SEC = 1.0
CHEAP_PREVIEW_MIN_TAIL_SEC = 0.5

#: The ``chunk_policy`` setting: "auto" (from the engine, the default) or
#: "manual" (the chunk_* numbers in the settings file, verbatim).
AUTO = "auto"
MANUAL = "manual"


@dataclass(frozen=True)
class LivePlan:
    """The shape of one live dictation."""

    policy: ChunkPolicy
    preview_min_tail_sec: float
    #: "cheap" or "local", for the developer console: which row was chosen.
    row: str


def is_cheap(caps: EngineCaps) -> bool:
    return caps.cost.fixed_sec < CHEAP_CALL_SEC


def plan_for(caps: EngineCaps, trailing_silence_sec: float = 0.6) -> LivePlan:
    """The live shape for an engine with these capabilities."""
    cheap = is_cheap(caps)
    lo, soft, force = CHEAP_POLICY if cheap else LOCAL_POLICY
    return LivePlan(
        policy=ChunkPolicy(
            min_sec=lo, soft_max_sec=soft, force_cut_sec=force,
            trailing_silence_sec=trailing_silence_sec,
        ),
        preview_min_tail_sec=CHEAP_PREVIEW_MIN_TAIL_SEC if cheap else LOCAL_PREVIEW_MIN_TAIL_SEC,
        row="cheap" if cheap else "local",
    )


def plan_from_settings(get: Callable[[str], Any], caps: Optional[EngineCaps]) -> LivePlan:
    """The plan for this dictation: the engine's, unless settings say manual.

    *get* is ``Settings.get``. "auto" is the default on purpose: the settings
    file persists every default it was ever written with, so an install that
    predates this still holds the old 6 / 15 / 20 as literal numbers. Reading
    those verbatim would keep every existing user on the slow shape.
    """
    trailing = float(get("chunk_trailing_silence_sec") or 0.6)
    if str(get("chunk_policy") or AUTO) == MANUAL or caps is None:
        return LivePlan(
            policy=ChunkPolicy(
                min_sec=float(get("chunk_min_sec")),
                soft_max_sec=float(get("chunk_soft_max_sec")),
                force_cut_sec=float(get("chunk_force_cut_sec")),
                trailing_silence_sec=trailing,
            ),
            preview_min_tail_sec=float(
                get("preview_min_tail_sec") or LOCAL_PREVIEW_MIN_TAIL_SEC
            ),
            row="manual",
        )
    return plan_for(caps, trailing)
