"""PHI de-identification for training data before it leaves the device.

This is the safety-critical gate of the cloud subsystem. Nothing is uploaded
until it has passed through :class:`DeIdentifier` and ``validate_clean`` has
confirmed no known patient identifier survives. The design is deliberately
conservative: when in doubt, a record is dropped rather than risk a breach.

Two surfaces are scrubbed:
  * **Text** — the correction strings and any context (names, IDs, dates).
  * **Audio** — segments whose transcript contains PHI are silenced, using the
    timestamps already produced by ``Transcriber.transcribe``.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from src.cloud.exceptions import PrivacyError

logger = logging.getLogger(__name__)

# Generic PHI patterns applied regardless of the known patient_info fields.
_DATE_PATTERNS = [
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    re.compile(
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
        r"Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b",
        re.IGNORECASE,
    ),
]
# Accession / MRN / NHS-style identifiers: optional letter prefix + 6–10 digits.
_ID_PATTERN = re.compile(r"\b[A-Z]{0,3}\d{6,10}\b")

# Fields of patient_info that are treated as direct identifiers.
_PHI_FIELDS = ("name", "id", "dob", "study_date", "referring", "accession")


class DeIdentifier:
    """Removes patient identifiers from training text and audio."""

    def __init__(self, patient_info: Optional[Dict[str, str]] = None) -> None:
        self._patient_info = patient_info or {}
        # Pre-compile word-boundary patterns for each non-empty identifier so we
        # can both redact and later validate against the same set.
        self._identifier_patterns: List[re.Pattern] = []
        for field in _PHI_FIELDS:
            value = (self._patient_info.get(field) or "").strip()
            if len(value) >= 2:
                self._identifier_patterns.append(
                    re.compile(re.escape(value), re.IGNORECASE)
                )

    # ------------------------------------------------------------------
    # Text
    # ------------------------------------------------------------------

    def deidentify_text(self, text: str) -> str:
        """Return *text* with patient identifiers, dates, and IDs redacted."""
        if not text:
            return text
        scrubbed = text
        for pat in self._identifier_patterns:
            scrubbed = pat.sub("[REDACTED]", scrubbed)
        for pat in _DATE_PATTERNS:
            scrubbed = pat.sub("[DATE]", scrubbed)
        scrubbed = _ID_PATTERN.sub("[ID]", scrubbed)
        return scrubbed

    def validate_clean(self, text: str) -> bool:
        """Return True only if no known patient identifier remains verbatim.

        Raises :class:`PrivacyError` if a known identifier survived — callers
        should treat the record as un-uploadable.
        """
        for pat in self._identifier_patterns:
            if pat.search(text):
                raise PrivacyError(
                    "Patient identifier survived de-identification; record dropped."
                )
        return True

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------

    def deidentify_audio(
        self,
        wav_path: str,
        segments: List[dict],
        out_dir: Path,
        session_id: str,
    ) -> Optional[str]:
        """Silence any audio segments whose transcript contains PHI.

        Returns the path to the scrubbed WAV clip, or None if the audio could
        not be processed (the caller then falls back to a text-only record).
        Requires soundfile + numpy, which are already core dependencies.
        """
        try:
            import soundfile as sf
        except Exception as exc:  # pragma: no cover - deps always present in app
            logger.warning("Audio de-identification skipped (deps missing): %s", exc)
            return None

        try:
            audio, sr = sf.read(wav_path, dtype="float32")
        except Exception as exc:
            logger.warning("Could not read audio for de-identification: %s", exc)
            return None

        if audio.ndim > 1:
            audio = audio[:, 0]

        pad = int(0.5 * sr)  # silence ±0.5s around a flagged segment
        for seg in segments:
            seg_text = (seg.get("text") or "")
            if self._contains_phi(seg_text):
                start = max(0, int(float(seg.get("start", 0)) * sr) - pad)
                end = min(len(audio), int(float(seg.get("end", 0)) * sr) + pad)
                audio[start:end] = 0.0
                logger.info("Silenced PHI audio segment %.1f–%.1fs",
                            seg.get("start", 0), seg.get("end", 0))

        out_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(wav_path.encode("utf-8")).hexdigest()[:8]
        out_path = out_dir / f"{session_id}_{digest}.wav"
        try:
            sf.write(str(out_path), audio, sr)
        except Exception as exc:
            logger.warning("Could not write de-identified audio: %s", exc)
            return None
        return str(out_path)

    def _contains_phi(self, text: str) -> bool:
        if not text:
            return False
        return any(pat.search(text) for pat in self._identifier_patterns)
