"""Pure decisions both live loops make — no state, no I/O, no Qt, no sockets.

The desktop worker and the web session each own a loop, and each has to answer
the same four questions: what prompt do we prime the decoder with, how sure was
the decoder, is this clip silent, and is the preview still worth its price.
Those answers live here so neither loop is the other's dependency — the desktop
used to import them from the web session's module, which read backwards and
made a front-end look like core.
"""

from __future__ import annotations

import numpy as np

from src.dictation.asr.types import AsrResult
from src.dictation.transcriber import RADIOLOGY_PROMPT
from src.features.adaptive_learning import get_custom_prompt_suffix


def build_context_prompt() -> str:
    """Return the initial prompt: base radiology vocab + learned terms.

    Committed text is intentionally NOT appended — doing so caused Whisper to
    echo prior words back into the current chunk under the small live beam.
    Chunks never overlap in this design, so there is no boundary-dedup step to
    lean on instead; the prompt just stays fixed.

    **Order is priority.** Whisper's prompt slot holds 223 tokens and it keeps
    the LAST 223, discarding the front without a word. The curated radiology
    vocabulary is sized to fit that slot on its own, so it goes last and always
    survives. The learned terms go in front, where they fill whatever room is
    left and are the ones dropped when there is none. Putting them last
    instead — as this did — let a full custom vocabulary (capped at 80 terms,
    about 216 tokens) push almost the entire shipped dictionary out of the
    decoder, which is invisible from the outside.
    """
    if custom_terms := get_custom_prompt_suffix():
        return f"{custom_terms} {RADIOLOGY_PROMPT}"
    return RADIOLOGY_PROMPT


def mean_confidence(result: AsrResult) -> float | None:
    """Mean word confidence across every segment that reported one.

    ``None`` when the engine gave no word timestamps for this call — distinct
    from 0.0 so the ledger's confidence gate never mistakes "no signal" for
    "the model was certain this is wrong".
    """
    vals = [c for seg in result.segments if (c := seg.confidence) is not None]
    return sum(vals) / len(vals) if vals else None


def rms(clip: np.ndarray) -> float:
    """Loudness of one clip. Accumulates in float64 because a long clip of
    float32 squares loses enough precision to move the silence decision."""
    if clip.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(clip, dtype=np.float64))))


def should_skip_preview(
    open_tail_sec: float, decode_cost: float, max_lag_sec: float
) -> bool:
    """Whether to drop this cycle's live preview decode.

    Compares what the preview would actually cost — ``open_tail_sec *
    decode_cost`` wall seconds, since the whole open tail is re-decoded — with
    *max_lag_sec*, the delay the radiologist is willing to accept before the
    words they just said appear. Costing more than that budget means the
    preview is showing stale words *and* holding up the committed chunks queued
    behind it, so it is dropped.

    A fast machine keeps its preview: at ``decode_cost`` 0.05 a 20-second tail
    costs 1 second, well inside a 3-second budget. A slow one loses it exactly
    when the tail has grown too expensive to be worth re-decoding.

    ``max_lag_sec <= 0`` turns the skip off entirely; ``decode_cost`` of 0.0
    means "not measured yet", which keeps the preview for the first cycles.

    The superseded test was ``decode_cost > 1.0 and open_tail_sec >
    max_lag_sec`` — "does this machine decode slower than speech?". It could
    not do the job for two reasons. This machine measures RTF 0.47-0.57
    (docs/dictation-accuracy.md), so the first clause was false and the preview
    was never skipped however far behind the loop fell. And ``decode_cost`` is
    total-wall-over-total-audio, which a fixed ~3.6s per-``transcribe()`` call
    cost (Whisper pads every clip to 30s) makes a function of *call count*
    rather than of throughput — so it rose above 1.0 only when the decoded
    clips were short, i.e. exactly when the preview was cheapest. What matters
    is this preview's own price, which is what this now asks.
    """
    if max_lag_sec <= 0:
        return False
    return open_tail_sec * decode_cost > max_lag_sec
