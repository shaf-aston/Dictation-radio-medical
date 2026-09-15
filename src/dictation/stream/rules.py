"""Pure decisions both live loops make: no state, no I/O, no Qt, no sockets.

The desktop worker and the web session each own a loop, and each has to answer
the same four questions: what prompt do we prime the decoder with, how sure was
the decoder, is this clip silent, and is the preview still worth its price.
Those answers live here so neither loop is the other's dependency: the desktop
used to import them from the web session's module, which read backwards and
made a front-end look like core.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.dictation.asr.types import AsrResult
from src.dictation.stream.vad import SpeechMark
from src.dictation.asr.prompt import RADIOLOGY_PROMPT
from src.features.adaptive_learning import get_custom_prompt_suffix


def build_context_prompt() -> str:
    """Return the initial prompt: base radiology vocab + learned terms.

    Committed text is intentionally NOT appended: doing so caused Whisper to
    echo prior words back into the current chunk under the small live beam.
    Chunks never overlap in this design, so there is no boundary-dedup step to
    lean on instead; the prompt just stays fixed.

    **Order is priority.** Whisper's prompt slot holds 223 tokens and it keeps
    the LAST 223, discarding the front without a word. The curated radiology
    vocabulary is sized to fit that slot on its own, so it goes last and always
    survives. The learned terms go in front, where they fill whatever room is
    left and are the ones dropped when there is none. Putting them last
    instead, as this did, let a full custom vocabulary (capped at 80 terms,
    about 216 tokens) push almost the entire shipped dictionary out of the
    decoder, which is invisible from the outside.
    """
    if custom_terms := get_custom_prompt_suffix():
        return f"{custom_terms} {RADIOLOGY_PROMPT}"
    return RADIOLOGY_PROMPT


def mean_confidence(result: AsrResult) -> float | None:
    """Mean word confidence across every segment that reported one.

    ``None`` when the engine gave no word timestamps for this call: distinct
    from 0.0 so the ledger's confidence gate never mistakes "no signal" for
    "the model was certain this is wrong".
    """
    vals = [c for seg in result.segments if (c := seg.confidence) is not None]
    return sum(vals) / len(vals) if vals else None


def low_confidence_words(result: AsrResult, ceiling: float) -> set:
    """The words in *result* the decoder itself was unsure about.

    Returned lower-cased and stripped of surrounding punctuation, because the
    front-end matches them against the finished report -- which the correction
    pipeline has since capitalised and punctuated. Matching whole words this
    way marks every later occurrence of the same word too, which is the
    conservative direction: the point is to draw the radiologist's eye to a
    word the machine guessed at, not to make a claim about one position.

    Empty when the engine gave no word timestamps: no signal is not the same
    as "the model was sure", and inventing marks would train the radiologist
    to ignore them.
    """
    out = set()
    for seg in result.segments:
        for word in seg.words:
            if word.confidence >= ceiling:
                continue
            cleaned = word.text.strip().strip(".,;:!?()[]{}\"'").lower()
            if cleaned:
                out.add(cleaned)
    return out


def level_db(clip: np.ndarray) -> float:
    """How loud a clip is, in dB below full scale (0 is as loud as the
    microphone can carry, -60 is near silence). For the developer console:
    a dictation that came back empty is either quiet or was never spoken, and
    this number is what tells the two apart. Accumulates in float64 because a
    long clip of float32 squares loses precision."""
    if clip.size == 0:
        return -120.0
    root_mean_square = float(np.sqrt(np.mean(np.square(clip, dtype=np.float64))))
    return round(20 * np.log10(max(root_mean_square, 1e-6)), 1)


def has_speech(marks: Sequence[SpeechMark], start: int, end: int) -> bool:
    """Does any VAD speech mark overlap ``[start, end)``?

    This is the only "is there anything to decode?" test in both live loops,
    and it is the VAD's verdict on purpose. The loudness gate it replaced
    (2026-09-06) dropped a quiet talker: measured on the tts set, Silero still
    hears speech at -46 dB, but an RMS floor of 0.002 called everything under
    -40 dB silence, and each dropped clip then fed the ambient estimate and
    raised the bar further. Two detectors disagreeing means one of them is
    redundant, and the VAD is the one the segmenter already trusts to cut on.
    """
    return any(m.start_sample < end and m.end_sample > start for m in marks)


def should_skip_preview(preview_cost_sec: float, max_lag_sec: float) -> bool:
    """Whether to drop this cycle's live preview decode.

    *preview_cost_sec* is what the last preview decode actually took, in wall
    seconds. *max_lag_sec* is the delay the radiologist will accept before the
    words they just said appear. Costing more than that budget means the
    preview is showing stale words *and* holding up the committed chunks queued
    behind it, so it is dropped and the last stable preview stays on screen.

    ``max_lag_sec <= 0`` turns the skip off entirely; a cost of 0.0 means "not
    measured yet", which keeps the preview for the first cycles.

    Two superseded tests are worth recording, because both were wrong in the
    same direction -- they modelled the price of a decode as a function of how
    much audio it covered:

    * ``decode_cost > 1.0 and open_tail_sec > max_lag_sec`` -- "does this
      machine decode slower than speech?". This machine measures RTF 0.47-0.57
      (docs/dictation-accuracy.md), so the first clause was never true and the
      preview was never skipped however far behind the loop fell.
    * ``open_tail_sec * decode_cost > max_lag_sec`` -- "what does re-decoding
      the whole open tail cost?". Measured on this machine, one ``transcribe()``
      call on the live model costs about the same whatever it is given:
      1.33s for a 3s clip, 1.36s for 6s, 1.52s for 25s. Whisper pads every clip
      to a 30-second window, so the encoder does identical work each time and
      the decoder's share is small. Multiplying by the tail length therefore
      grew a number that in reality stayed flat, and the preview was switched
      off part-way through every chunk -- exactly the stretch where the
      radiologist has said the most and can see the least.

    So the question is not how long the tail is. It is how long the last one of
    these calls took.
    """
    if max_lag_sec <= 0:
        return False
    return preview_cost_sec > max_lag_sec
