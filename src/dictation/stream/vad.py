"""Voice-activity detection — thin wrapper over faster-whisper's bundled Silero VAD.

No new dependency: faster_whisper ships ``silero_vad.onnx`` and this helper,
the same one ``WhisperModel(vad_filter=True)`` already uses internally.
Exposing it directly lets the segmenter cut chunks at real silence instead of
decoding fixed-size windows blind to where the radiologist actually paused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

#: Silero VAD is hardwired to 16kHz inside faster_whisper; the recorder
#: (src/dictation/audio.py) already captures at this rate, so no resampling
#: seam exists anywhere in this pipeline.
SAMPLE_RATE = 16000


@dataclass(frozen=True)
class SpeechMark:
    """One speech region, in samples at :data:`SAMPLE_RATE`."""

    start_sample: int
    end_sample: int


def detect_speech(
    audio: np.ndarray,
    min_silence_ms: int = 300,
    speech_pad_ms: int = 200,
) -> List[SpeechMark]:
    """Return speech-only sample ranges in *audio* (mono float32 @ 16kHz).

    ``min_silence_ms`` is deliberately shorter than faster-whisper's own
    ``vad_filter=True`` default (2000ms, tuned to filter non-speech noise out
    of a whole-file decode) — the segmenter needs to catch a radiologist's
    ordinary between-sentence pause (typically 300-800ms) as a cut point, not
    just long silences.
    """
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    options = VadOptions(
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
    )
    marks = get_speech_timestamps(audio, options)
    return [SpeechMark(m["start"], m["end"]) for m in marks]
