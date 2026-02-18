from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain-specific initial prompt fed to Whisper to prime its vocabulary.
# This dramatically improves accuracy for MSK radiology without any extra model.
# ---------------------------------------------------------------------------
_MSK_INITIAL_PROMPT = (
    "Musculoskeletal radiology report. "
    "ACL, PCL, MCL, LCL, UCL, TFCC, SLIL, ATFL, CFL, SLAP, HAGL. "
    "Supraspinatus, infraspinatus, subscapularis, teres minor. "
    "Glenohumeral, acromioclavicular, patellofemoral, tibiofemoral, sacroiliac. "
    "Meniscus, meniscal, chondromalacia, osteochondral, subchondral, articular cartilage. "
    "Kellgren-Lawrence, Outerbridge, Modic, ARCO, SPARCC. "
    "Tendinopathy, tenosynovitis, bursitis, synovitis, osteophyte, osteophytosis. "
    "Spondylolisthesis, spondylolysis, foraminal stenosis, disc protrusion, disc extrusion. "
    "T1-weighted, T2-weighted, STIR, PDFS, FLAIR, DWI, ADC. "
    "Hyperintense, hypointense, isointense, bone marrow oedema, avascular necrosis. "
    "Fat-saturated, fat-suppressed, proton density-weighted. "
    "Millimetres, centimetres, degrees. "
    "Technique: Findings: Impression: "
)

# Supported model sizes in order of speed (fastest first).
# Users may choose based on hardware / accuracy requirements.
SUPPORTED_MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]


class Transcriber:
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
        self._model: Optional[WhisperModel] = None

    # ------------------------------------------------------------------
    # Model loading with compute-type fallback chain
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

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
        audio_path: str,
        language: str = "en",
        vad_filter: bool = True,
        beam_size: Optional[int] = None,
        initial_prompt: Optional[str] = None,
    ) -> Tuple[str, List[Dict]]:
        """
        Transcribe *audio_path* and return (full_text, segment_list).

        segment_list items: {"start": float, "end": float, "text": str}

        initial_prompt overrides the built-in MSK prompt when supplied.
        Pass initial_prompt="" to disable prompting entirely.
        """
        self._ensure_model()
        beam_size = beam_size if beam_size is not None else 5

        # Choose prompt
        if initial_prompt is None:
            prompt = _MSK_INITIAL_PROMPT if self.use_msk_prompt else None
        else:
            prompt = initial_prompt or None  # empty string → no prompt

        kwargs = dict(
            language=language,
            vad_filter=vad_filter,
            beam_size=beam_size,
            condition_on_previous_text=True,
            initial_prompt=prompt,
        )

        try:
            segments_gen, _ = self._model.transcribe(audio_path, **kwargs)
        except Exception:
            # VAD may fail on some systems – retry without it
            kwargs["vad_filter"] = False
            segments_gen, _ = self._model.transcribe(audio_path, **kwargs)

        # Materialise the lazy generator and build output structures
        parts: List[str] = []
        seg_list: List[Dict] = []
        prev_end: Optional[float] = None

        for seg in segments_gen:
            seg_text = (seg.text or "").strip()
            seg_list.append({"start": seg.start, "end": seg.end, "text": seg.text})

            if not seg_text:
                prev_end = seg.end
                continue

            if prev_end is not None:
                gap = (seg.start or 0.0) - prev_end
                parts.append("\n" if gap >= 2.5 else " ")

            parts.append(seg_text)
            prev_end = seg.end

        return "".join(parts).strip(), seg_list
