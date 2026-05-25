"""Whisper transcription wrapper tuned for radiology dictation.

Lazy-loads `faster-whisper` on first transcribe() to keep startup instant.
Applies a domain-specific initial prompt (`_RADIOLOGY_INITIAL_PROMPT`, loaded
from src/dictation/resources/radiology_prompt.txt) to prime the model's
vocabulary and filters per-segment hallucinations before returning text.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from src.features.file_manager import radiology_prompt_path

logger = logging.getLogger(__name__)


def _load_radiology_prompt() -> str:
    """Read the prompt file: drop blank/comment lines and collapse to one line.

    Whisper truncates from the front of the prompt, so order matters — file
    content is preserved verbatim except for whitespace normalisation.
    """
    raw = radiology_prompt_path().read_text(encoding="utf-8")
    lines = [ln.strip() for ln in raw.splitlines()]
    return " ".join(ln for ln in lines if ln and not ln.startswith("#"))


# Built once at import. The file is tiny (~3KB) so the I/O is negligible, and
# downstream callers (workers/, tests/) treat this as a plain string constant.
_RADIOLOGY_INITIAL_PROMPT = _load_radiology_prompt()

# Supported model sizes in order of speed (fastest first).
SUPPORTED_MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]


class Transcriber:
    """Wraps faster-whisper for radiology transcription.

    The model loads lazily on the first `transcribe()` call so construction
    is instant. The compute-type fallback chain (int8 → float32) selects the
    fastest backend the host CPU/GPU supports.

    Args:
        model_size: One of `SUPPORTED_MODELS` (default `'base'`).
        device: `'auto'`, `'cpu'`, or `'cuda'`.
        compute_type: ctranslate2 quantisation; auto-selected if None.
        use_msk_prompt: Inject the radiology initial prompt before each
            transcription.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: Optional[str] = None,
        use_msk_prompt: bool = True,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.use_msk_prompt = use_msk_prompt
        self._model: Optional[Any] = None  # WhisperModel, loaded lazily

    # ------------------------------------------------------------------
    # Model loading with compute-type fallback chain
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        
        # Lazy import: only load heavy ctranslate2 dependency when first needed
        from faster_whisper import WhisperModel

        compute_types = (
            [self.compute_type, "int8", "float32"]
            if self.compute_type
            else ["int8", "float32"]
        )

        last_err: Optional[Exception] = None
        for ct in compute_types:
            if ct is None:
                continue
            try:
                logger.info(
                    "Loading Whisper model: %s  device=%s  compute_type=%s",
                    self.model_size, self.device, ct,
                )
                self._model = WhisperModel(
                    self.model_size, device=self.device, compute_type=ct
                )
                self.compute_type = ct
                logger.info("Model loaded successfully with compute_type=%s", ct)
                return
            except Exception as exc:
                logger.warning("Failed to load with compute_type=%s: %s", ct, exc)
                last_err = exc

        raise RuntimeError(
            f"Failed to load Whisper model '{self.model_size}'. Last error: {last_err}"
        )

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: Any,
        language: str = "en",
        vad_filter: bool = True,
        beam_size: Optional[int] = None,
        initial_prompt: Optional[str] = None,
        pause_threshold: float = 2.5,
        condition_on_previous_text: bool = True,
    ) -> Tuple[str, List[Dict]]:
        """
        Transcribe *audio* and return (full_text, segment_list).

        *audio* may be a file path (str), a BinaryIO stream, or a numpy
        ndarray of float32 samples.  Passing an ndarray avoids the ffmpeg
        decode step, which is faster for WAV data already in memory.

        segment_list items: {"start": float, "end": float, "text": str}

        initial_prompt overrides the built-in radiology prompt when supplied.
        Pass initial_prompt="" to disable prompting entirely.
        """
        self._ensure_model()
        assert self._model is not None
        beam_size = beam_size if beam_size is not None else 5

        # Choose prompt
        if initial_prompt is None:
            prompt = _RADIOLOGY_INITIAL_PROMPT if self.use_msk_prompt else None
        else:
            prompt = initial_prompt or None  # empty string → no prompt

        kwargs = dict(
            language=language,
            vad_filter=vad_filter,
            beam_size=beam_size,
            condition_on_previous_text=condition_on_previous_text,
            initial_prompt=prompt,
            # ── Hallucination / quality thresholds ──
            # Segments with compression_ratio above this are likely repetitive
            # hallucinations (e.g. "the the the the …").
            compression_ratio_threshold=2.4,
            # Segments with average log-prob below this are low-confidence and
            # often garbage output from silence or noise.
            log_prob_threshold=-1.0,
            # Probability that a segment is "no speech" – higher = stricter
            # filtering of silence / background noise.
            no_speech_threshold=0.6,
            # Temperature fallback: start with greedy (0.0) for deterministic
            # output; if that fails the quality checks, retry with increasing
            # temperature for diversity.  This is Whisper's built-in fallback.
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8],
        )

        try:
            segments_gen, info = self._model.transcribe(audio, **kwargs)
        except Exception:
            # VAD may fail on some systems – retry without it
            kwargs["vad_filter"] = False
            segments_gen, info = self._model.transcribe(audio, **kwargs)

        # Materialise the lazy generator and build output structures
        parts: List[str] = []
        seg_list: List[Dict] = []
        prev_end: Optional[float] = None

        for seg in segments_gen:
            seg_text = (seg.text or "").strip()

            # ── Skip hallucinated / low-quality segments ──
            if seg_text and _is_hallucination(seg_text, seg):
                logger.debug("Filtered hallucination: %r", seg_text)
                prev_end = seg.end
                continue

            seg_list.append({"start": seg.start, "end": seg.end, "text": seg.text})

            if not seg_text:
                prev_end = seg.end
                continue

            if prev_end is not None:
                gap = (seg.start or 0.0) - prev_end
                parts.append("\n" if gap >= pause_threshold else " ")

            parts.append(seg_text)
            prev_end = seg.end

        return "".join(parts).strip(), seg_list


# ---------------------------------------------------------------------------
# Hallucination detection
# ---------------------------------------------------------------------------

# Common Whisper hallucination phrases that appear when the model decodes
# silence, noise, or very faint audio.
_HALLUCINATION_PHRASES = {
    "thank you for watching",
    "thanks for watching",
    "thank you for listening",
    "please subscribe",
    "subscribe to",
    "like and subscribe",
    "see you next time",
    "see you in the next",
    "bye bye",
    "goodbye",
    "thank you so much",
    "thanks for joining",
    "have a great day",
    "subtitles by",
    "translated by",
    "amara.org",
    "you",
}


def _is_hallucination(text: str, seg: Any) -> bool:
    """Return True if a segment looks like a Whisper hallucination."""
    lower = text.lower().strip()

    # 1. Known hallucination phrases
    if lower in _HALLUCINATION_PHRASES:
        return True
    for phrase in _HALLUCINATION_PHRASES:
        if lower.startswith(phrase):
            return True

    # 2. Very short text on a long segment (Whisper filling silence)
    duration = getattr(seg, "end", 0) - getattr(seg, "start", 0)
    if len(lower.split()) <= 2 and duration > 8.0:
        return True

    # 3. Excessive repetition within a single segment
    words = lower.split()
    if len(words) >= 6:
        unique = set(words)
        if len(unique) <= 2:
            return True

    # 4. Low average log probability (if available)
    avg_logprob = getattr(seg, "avg_logprob", None)
    if avg_logprob is not None and avg_logprob < -1.5:
        no_speech = getattr(seg, "no_speech_prob", 0.0)
        if no_speech > 0.4:
            return True

    return False
