"""Whisper transcription wrapper tuned for radiology dictation.

Lazy-loads `faster-whisper` on first transcribe() to keep startup instant.
Model instances are shared process-wide per (model, device, compute_type) —
which is what makes startup warmup (src/dictation/warmup.py) effective for the
recording worker.
Applies a domain-specific initial prompt (`RADIOLOGY_PROMPT`, loaded
from src/dictation/resources/radiology_prompt.txt) to prime the model's
vocabulary and filters per-segment hallucinations before returning text.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.features.file_manager import radiology_prompt_path, whisper_cache_dir

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
# downstream callers treat this as a plain string constant. It is the same
# radiology vocabulary for every engine, so it is re-exported from
# src/dictation/asr/ — code outside asr/ imports it from there, never from here.
RADIOLOGY_PROMPT = _load_radiology_prompt()

# Supported model sizes in order of speed (fastest first).
#
# The `.en` models are English-only. For English dictation they are both faster and
# more accurate than the same-sized multilingual model, because none of the capacity
# is spent on the other 98 languages — so they are the right default here and they
# have to be selectable. Leaving them out silently downgraded anyone whose settings
# named one: the picker ignored the unknown value and fell back to its first entry.
SUPPORTED_MODELS = [
    "tiny.en", "tiny",
    "base.en", "base",
    "small.en", "small",
    "medium.en", "medium",
    "large-v2", "large-v3",
]

# What to fall back to when a setting names a model this build does not know.
DEFAULT_MODEL = "base.en"


def resolve_model(name: Optional[str]) -> str:
    """Return *name* if this build supports it, else :data:`DEFAULT_MODEL`, loudly.

    Both front-ends used to drop an unrecognised model on the floor without a word —
    the web app reassigned it, and the desktop combo box ignored ``setCurrentText``
    for a value it had no item for and stayed on its first entry. Either way the
    radiologist got a different model from the one their settings named, with no
    hint that it had happened. Falling back is fine; doing it in silence is not.
    """
    if name in SUPPORTED_MODELS:
        return name  # type: ignore[return-value]
    logger.warning(
        "Model %r is not one of %s — using %s instead. Check 'model_size' in "
        "dictation_settings.json.", name, SUPPORTED_MODELS, DEFAULT_MODEL,
    )
    return DEFAULT_MODEL

# Process-wide model cache: (model_ref, device, requested compute_type-or-None)
# -> (WhisperModel, resolved compute_type). Loading a model takes multiple
# seconds and hundreds of MB; sharing one instance across Transcriber objects
# (warmup thread, recording worker, final pass) avoids paying that twice.
_MODEL_CACHE: Dict[Tuple[str, str, Optional[str]], Tuple[Any, str]] = {}
_MODEL_CACHE_LOCK = threading.Lock()
# LRU bound: each entry pins hundreds of MB for the process lifetime, so
# switching model size (or activating a fine-tune) must evict, not accumulate.
# 2 tolerates one warmup/worker mismatch without reload thrash; live callers
# holding an evicted model keep their own reference and are unaffected.
_MODEL_CACHE_MAX = 2


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
        model_path: Optional local CTranslate2 model directory (a fine-tuned
            model downloaded from Lightning AI). When set it overrides
            ``model_size`` — faster-whisper loads the model straight from disk.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: Optional[str] = None,
        use_msk_prompt: bool = True,
        model_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.use_msk_prompt = use_msk_prompt
        self.model_path = str(model_path) if model_path else None
        self._model: Optional[Any] = None  # WhisperModel, loaded lazily

    # ------------------------------------------------------------------
    # Model loading with compute-type fallback chain
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        t_start = time.time()
        # Lazy import: only load heavy ctranslate2 dependency when first needed
        from faster_whisper import WhisperModel

        # A fine-tuned model directory is passed verbatim to faster-whisper,
        # which accepts a local path in place of a named model size.
        model_ref = self.model_path or self.model_size

        # Stock models download into data/cache/whisper/ (not the hidden
        # per-user HuggingFace cache) so every cache lives in one deletable
        # place. Ignored by faster-whisper when model_ref is a local path.
        download_root = str(whisper_cache_dir())

        # Keyed on the *requested* compute_type (may be None) so two callers
        # asking for the same thing share one instance. The lock covers the
        # whole check-load-store: double loading wastes seconds and RAM, and
        # serialising loads is fine (warmup thread vs worker QThread race).
        cache_key = (model_ref, self.device, self.compute_type)
        with _MODEL_CACHE_LOCK:
            cached = _MODEL_CACHE.pop(cache_key, None)
            if cached is not None:
                _MODEL_CACHE[cache_key] = cached  # re-insert: LRU move-to-end
                self._model, self.compute_type = cached
                return

            compute_types = (
                [self.compute_type, "int8", "int8_float32", "float32"]
                if self.compute_type
                else ["int8", "int8_float32", "float32"]
            )

            # On CPU, CTranslate2 defaults to 4 intra-op threads regardless of core
            # count — leave the OS a couple of cores and use the rest.
            cpu_threads = 0 if self.device == "cuda" else max(4, (os.cpu_count() or 4) - 2)

            last_err: Optional[Exception] = None
            seen: set = set()
            for ct in compute_types:
                if ct is None or ct in seen:
                    continue
                seen.add(ct)
                try:
                    logger.info(
                        "Loading Whisper model: %s  device=%s  compute_type=%s  cpu_threads=%d",
                        model_ref, self.device, ct, cpu_threads,
                    )
                    self._model = WhisperModel(
                        model_ref, device=self.device, compute_type=ct,
                        cpu_threads=cpu_threads, download_root=download_root,
                    )
                    _MODEL_CACHE[cache_key] = (self._model, ct)
                    while len(_MODEL_CACHE) > _MODEL_CACHE_MAX:
                        evicted = next(iter(_MODEL_CACHE))
                        del _MODEL_CACHE[evicted]
                        logger.info("Evicted cached Whisper model %s", evicted)
                    self.compute_type = ct
                    elapsed = time.time() - t_start
                    logger.info("Model loaded successfully with compute_type=%s [%.2fs]", ct, elapsed)
                    return
                except (ValueError, RuntimeError) as exc:
                    logger.warning("Failed to load with compute_type=%s: %s", ct, exc)
                    last_err = exc

        raise RuntimeError(
            f"Failed to load Whisper model '{model_ref}'. Last error: {last_err}"
        )

    def preload(self) -> None:
        """Load the Whisper model now instead of on the first ``transcribe()``.

        Public warm-up seam: lets a front-end pay the multi-second model load at
        app startup on a background thread (see :mod:`src.dictation.warmup`), so
        the radiologist's first spoken chunk is not the thing that stalls.
        Idempotent — a no-op once the model is loaded.
        """
        self._ensure_model()

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
        temperature: Optional[Union[float, List[float]]] = None,
        word_timestamps: bool = False,
        hotwords: Optional[List[str]] = None,
    ) -> Tuple[str, List[Dict]]:
        """
        Transcribe *audio* and return (full_text, segment_list).

        *audio* may be a file path (str), a BinaryIO stream, or a numpy
        ndarray of float32 samples.  Passing an ndarray avoids the ffmpeg
        decode step, which is faster for WAV data already in memory.

        segment_list items: {"start": float, "end": float, "text": str}, plus
        a "words" key (list of {"text","start","end","probability"}) when
        word_timestamps=True — the confidence signal the AsrEngine port
        (src/dictation/asr/) surfaces to callers. Off by default: it costs a
        little extra decode time and nothing needs it until the confidence-
        gated post-processing pipeline (M4) consumes it.

        initial_prompt overrides the built-in radiology prompt when supplied.
        Pass initial_prompt="" to disable prompting entirely.
        """
        self._ensure_model()
        assert self._model is not None
        beam_size = beam_size if beam_size is not None else 5

        # Choose prompt
        if initial_prompt is None:
            prompt = RADIOLOGY_PROMPT if self.use_msk_prompt else None
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
            # Callers may override (e.g. the live cycle passes 0.0 for a
            # single greedy decode).
            temperature=temperature if temperature is not None else [0.0, 0.2, 0.4, 0.6, 0.8],
            word_timestamps=word_timestamps,
        )
        if hotwords:
            kwargs["hotwords"] = " ".join(hotwords)

        t_transcribe = time.time()
        try:
            segments_gen, info = self._model.transcribe(audio, **kwargs)
        except Exception as exc:
            logger.warning("Transcription failed (vad_filter=%s): %s — retrying without VAD",
                           kwargs.get("vad_filter"), exc)
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

            seg_entry: Dict[str, Any] = {"start": seg.start, "end": seg.end, "text": seg.text}
            seg_words = getattr(seg, "words", None)
            if seg_words:
                seg_entry["words"] = [
                    {"text": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                    for w in seg_words
                ]
            seg_list.append(seg_entry)

            if not seg_text:
                prev_end = seg.end
                continue

            if prev_end is not None:
                gap = (seg.start or 0.0) - prev_end
                parts.append("\n" if gap >= pause_threshold else " ")

            parts.append(seg_text)
            prev_end = seg.end

        elapsed = time.time() - t_transcribe
        logger.info("Transcription [%.2fs]", elapsed)
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

# Short phrases (<=2 words) are only hallucinations when they are the WHOLE
# segment — matching them as a prefix silently deleted real dictation like
# "Your report shows…" or "Young patient…". Long YouTube-outro phrases stay
# prefix-matched (they trail into varied garbage).
_SHORT_PHRASES = {p for p in _HALLUCINATION_PHRASES if len(p.split()) <= 2}
_LONG_PHRASES = _HALLUCINATION_PHRASES - _SHORT_PHRASES

# `(?!)` never matches — a safe alternation when a phrase set is empty (an
# empty `(?:)` would otherwise match every segment).
def _alt(phrases: set) -> str:
    return "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True)) or "(?!)"

_HALLUCINATION_RE = re.compile(r"^(?:" + _alt(_LONG_PHRASES) + r")", re.IGNORECASE)
# Whole-segment (trailing punctuation/whitespace tolerated) for short phrases.
_HALLUCINATION_EXACT_RE = re.compile(
    r"^(?:" + _alt(_SHORT_PHRASES) + r")[\s.,!?]*$", re.IGNORECASE
)


def _is_hallucination(text: str, seg: Any) -> bool:
    """Return True if a segment looks like a Whisper hallucination."""
    lower = text.lower().strip()

    # 1. Known hallucination phrases: long outros by prefix, short ones only
    #    when they are the entire segment (avoids deleting "your"/"young"…).
    if _HALLUCINATION_RE.match(lower) or _HALLUCINATION_EXACT_RE.match(lower):
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
