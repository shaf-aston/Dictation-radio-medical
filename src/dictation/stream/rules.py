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


class AdaptiveFloor:
    """Is this clip silence — judged against THIS room, not one fixed number.

    A single fixed loudness threshold cannot be right for two different
    radiologists: a quiet talker's real speech can sit below a number tuned
    for a normal voice, and gets thrown away before the decoder ever sees it;
    a noisy room's background hum can sit above that same number and get fed
    to the decoder as if it were speech, which is what the confidence gate
    calls the decoder's own hallucination anti-measure exists to catch
    upstream of. This tracks the room's own ambient level instead and gates
    relative to *it*.

    Only clips already judged quiet feed the estimate, so a loud sentence
    never drags its own gate up and locks out the next quiet word — the
    estimate follows the room, not the voice. ``floor_min`` is a hard safety
    net for literal digital silence, so the very first clip (before any
    estimate exists) is never mistaken for speech.
    """

    def __init__(self, floor_min: float, margin: float = 2.5, smoothing: float = 0.2):
        self.floor_min = max(0.0, float(floor_min))
        self.margin = max(1.0, float(margin))
        self.smoothing = min(1.0, max(0.0, float(smoothing)))
        self._estimate = self.floor_min

    @property
    def threshold(self) -> float:
        return max(self.floor_min, self._estimate * self.margin)

    def is_silence(self, level: float) -> bool:
        quiet = level < self.threshold
        if quiet:
            self._estimate = (1 - self.smoothing) * self._estimate + self.smoothing * level
        return quiet


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
