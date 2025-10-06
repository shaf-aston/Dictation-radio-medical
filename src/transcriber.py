from __future__ import annotations
from typing import List, Dict, Tuple, Optional

from faster_whisper import WhisperModel


class Transcriber:
    def __init__(self, model_size: str = "base", device: str = "auto", compute_type: Optional[str] = None):
        self.model_size = model_size
        self.device = device
        # Defer compute_type decision to runtime with fallbacks
        self.compute_type = compute_type
        self._model: Optional[WhisperModel] = None

    def _ensure_model(self) -> None:
        if self._model is None:
            last_err: Optional[Exception] = None
            # Preferred order: try GPU-friendly types first, then portable CPU types
            candidates = [
                self.compute_type,
                "float16",
                "int8_float16",
                "int8",
                "int16",
                "float32",
            ]
            # Remove Nones and keep order while deduplicating
            seen = set()
            ct_list = []
            for ct in candidates:
                if ct is None:
                    continue
                if ct not in seen:
                    seen.add(ct)
                    ct_list.append(ct)

            for ct in ct_list:
                try:
                    self._model = WhisperModel(self.model_size, device=self.device, compute_type=ct)
                    self.compute_type = ct
                    break
                except Exception as e:
                    last_err = e
                    continue
            if self._model is None:
                raise RuntimeError(f"Failed to load Whisper model with compute types {ct_list}: {last_err}")

    def transcribe(self, audio_path: str, language: str = "en", vad_filter: bool = True) -> Tuple[str, List[Dict]]:
        self._ensure_model()
        try:
            segments, info = self._model.transcribe(
                audio_path,
                language=language,
                vad_filter=vad_filter,
                beam_size=5,
                condition_on_previous_text=True,
            )
        except Exception:
            # Retry once without VAD filtering (some backends may lack optional deps)
            segments, info = self._model.transcribe(
                audio_path,
                language=language,
                vad_filter=False,
                beam_size=5,
                condition_on_previous_text=True,
            )
        seg_list = []
        texts = []
        for s in segments:
            seg_list.append({"start": s.start, "end": s.end, "text": s.text})
            texts.append(s.text)
        full_text = " ".join(texts).strip()
        return full_text, seg_list
