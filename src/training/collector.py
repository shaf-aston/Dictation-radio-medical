"""Captures correction triples during dictation for later cloud fine-tuning.

The collector is the single bridge between the local pipeline and the cloud
subsystem. It is intentionally one-directional: ``adaptive_learning`` and
``voice_commands`` call *into* the collector, but never import cloud modules.
This keeps the offline pipeline free of any cloud dependency.

Everything here is a no-op unless BOTH ``cloud_enabled`` and
``cloud_training_consent`` are true in settings — so with default settings the
app retains no audio and stages no data.

Lifecycle within one dictation session:
    start_session(session_id, wav_path, patient_info)
      → update_segments(segments)        # called each transcription cycle
      → record_text_correction(w, c)     # called when the user corrects output
    prepare_audio()                       # de-identify audio while the WAV exists
    ... user reviews transcript, typed corrections still flow in ...
    finalize_session()                    # persist records (next recording / close)

Why two phases: the temp WAV is deleted right after transcription, but the
user corrects spelling *during review*, which happens later. We de-identify the
audio up front (``prepare_audio``) while the file is still on disk, then keep the
session open so review-time corrections are still captured, and persist
everything at the next boundary (a new recording, or app close).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class _PendingCorrection:
    wrong: str
    correct: str
    ts_start: Optional[float] = None
    ts_end: Optional[float] = None


@dataclass
class _Session:
    session_id: str
    wav_path: Optional[str]
    patient_info: Dict[str, str]
    model_version: str
    accent_profile: str
    segments: List[dict] = field(default_factory=list)
    corrections: List[_PendingCorrection] = field(default_factory=list)
    audio_clip: Optional[str] = None
    audio_prepared: bool = False


class CorrectionCollector:
    """Collects and stages correction triples, gated on user consent."""

    _instance: Optional["CorrectionCollector"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "CorrectionCollector":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        self._session: Optional[_Session] = None
        self._session_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Consent gate
    # ------------------------------------------------------------------

    @staticmethod
    def _consent_active() -> bool:
        """True only when the user has opted in to cloud training."""
        try:
            from src.core.settings import Settings
            s = Settings()
            return bool(s.get("cloud_enabled")) and bool(s.get("cloud_training_consent"))
        except Exception as exc:
            logger.debug("Could not read consent settings; treating as disabled: %s", exc)
            return False

    def is_enabled(self) -> bool:
        return self._consent_active()

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start_session(
        self,
        session_id: str,
        wav_path: Optional[str],
        patient_info: Optional[Dict[str, str]] = None,
        model_version: str = "base",
        accent_profile: str = "neutral",
    ) -> None:
        """Begin capturing for a recording session. No-op without consent."""
        if not self._consent_active():
            return
        with self._session_lock:
            self._session = _Session(
                session_id=session_id,
                wav_path=wav_path,
                patient_info=patient_info or {},
                model_version=model_version,
                accent_profile=accent_profile,
            )
        logger.info("Correction capture started for session %s", session_id)

    def update_segments(self, segments: List[dict]) -> None:
        """Merge the latest window's segments into the session's full timeline.

        The worker only transcribes a sliding window, so each emission covers
        recent audio. We merge by absolute start time (rounded to 0.1s) so the
        accumulated list spans the whole recording — important for audio
        de-identification, where PHI is often spoken near the start.
        """
        if not segments:
            return
        with self._session_lock:
            if self._session is None:
                return
            by_start = {round(float(s.get("start", 0)), 1): s
                        for s in self._session.segments}
            for s in segments:
                by_start[round(float(s.get("start", 0)), 1)] = s
            self._session.segments = [by_start[k] for k in sorted(by_start)]

    def is_session_captured(self, wav_path: Optional[str]) -> bool:
        """True if *wav_path* belongs to the active capture session.

        ``recording_session`` consults this to decide whether to preserve the
        temp WAV (needed for audio de-identification) instead of deleting it.
        """
        with self._session_lock:
            return (
                self._session is not None
                and wav_path is not None
                and self._session.wav_path == wav_path
            )

    # ------------------------------------------------------------------
    # Correction capture
    # ------------------------------------------------------------------

    def record_text_correction(self, wrong: str, correct: str) -> None:
        """Record a (wrong → correct) edit, time-locating it via segments.

        Called from ``adaptive_learning.learn_correction`` and the
        ``voice_commands`` correction hook. Safe to call with no active
        session — it simply returns.
        """
        if not self._consent_active():
            return
        wrong = (wrong or "").strip()
        correct = (correct or "").strip()
        if not wrong or not correct or wrong.lower() == correct.lower():
            return
        with self._session_lock:
            if self._session is None:
                return
            ts_start, ts_end = self._locate(wrong, self._session.segments)
            self._session.corrections.append(
                _PendingCorrection(wrong=wrong, correct=correct,
                                   ts_start=ts_start, ts_end=ts_end)
            )
        logger.debug("Captured correction candidate: %r → %r", wrong, correct)

    @staticmethod
    def _locate(word: str, segments: List[dict]) -> tuple:
        """Find the (start, end) of the segment containing *word*, if any."""
        wl = word.lower()
        return next(
            (
                (float(seg.get("start", 0.0)), float(seg.get("end", 0.0)))
                for seg in segments
                if wl in (seg.get("text") or "").lower()
            ),
            (None, None),
        )

    # ------------------------------------------------------------------
    # Finalisation — de-identify and persist (two phases)
    # ------------------------------------------------------------------

    def prepare_audio(self) -> None:
        """De-identify the session audio while the temp WAV still exists.

        Called at the end of transcription, *before* the WAV is deleted. The
        resulting clip is held on the open session so that corrections typed
        during review can still be persisted against it later. Idempotent and a
        no-op without consent or audio.
        """
        if not self._consent_active():
            return
        with self._session_lock:
            session = self._session
            if (session is None or session.audio_prepared
                    or not session.wav_path):
                if session is not None:
                    session.audio_prepared = True
                return
            wav_path, patient_info = session.wav_path, session.patient_info
            segments = list(session.segments)
            session_id = session.session_id

        # Run de-identification outside the lock (file I/O), then store back.
        clip = self._deidentify_audio(patient_info, wav_path, segments, session_id)
        with self._session_lock:
            if self._session is not None and self._session.session_id == session_id:
                self._session.audio_clip = clip
                self._session.audio_prepared = True

    @staticmethod
    def _deidentify_audio(patient_info, wav_path, segments, session_id):
        from src.cloud.privacy import DeIdentifier
        from src.features.file_manager import training_audio_dir
        return DeIdentifier(patient_info).deidentify_audio(
            wav_path, segments, training_audio_dir(), session_id,
        )

    def finalize_session(self) -> int:
        """Persist the session's corrections and close it. Returns count saved."""
        if not self._consent_active():
            self._clear()
            return 0
        with self._session_lock:
            session = self._session
            self._session = None
        if session is None or not session.corrections:
            return 0
        return self._persist(session)

    def _persist(self, session: _Session) -> int:
        from src.cloud.exceptions import PrivacyError
        from src.cloud.privacy import DeIdentifier
        from src.features import audit_log
        from src.training.schemas import CorrectionRecord
        from src.training.staging_db import get_staging_db

        deid = DeIdentifier(session.patient_info)

        # Audio is normally de-identified up front by prepare_audio (while the
        # WAV existed). Fall back to doing it now if that never ran and the file
        # is somehow still present.
        audio_clip = session.audio_clip
        if audio_clip is None and not session.audio_prepared and session.wav_path:
            audio_clip = self._deidentify_audio(
                session.patient_info, session.wav_path,
                session.segments, session.session_id,
            )

        db = get_staging_db()
        saved = 0
        now = datetime.now(timezone.utc).isoformat()
        for corr in session.corrections:
            clean_wrong = deid.deidentify_text(corr.wrong)
            clean_correct = deid.deidentify_text(corr.correct)
            try:
                deid.validate_clean(clean_wrong)
                deid.validate_clean(clean_correct)
            except PrivacyError:
                audit_log.log_training_record_dropped("phi_validation_failed")
                continue
            record = CorrectionRecord(
                session_id=session.session_id,
                created_at=now,
                wrong_text=clean_wrong,
                correct_text=clean_correct,
                audio_path=audio_clip,
                ts_start=corr.ts_start,
                ts_end=corr.ts_end,
                model_version=session.model_version,
                accent_profile=session.accent_profile,
                deidentified=True,
            )
            db.insert_correction(record)
            saved += 1
        logger.info("Persisted %d/%d corrections for session %s",
                    saved, len(session.corrections), session.session_id)
        return saved

    def _clear(self) -> None:
        with self._session_lock:
            self._session = None

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_pending_count(self) -> int:
        """Total de-identified corrections staged and awaiting upload."""
        if not self._consent_active():
            return 0
        try:
            from src.training.staging_db import get_staging_db
            return get_staging_db().pending_count()
        except Exception as exc:
            logger.warning("Could not read pending count: %s", exc)
            return 0

    def capture_image_label(self, record) -> tuple[bool, str]:
        """Stage a labeled image for training.

        Args:
            record: ImageLabelRecord with image_path, labels, and consent_flags.

        Returns:
            (True, message) on success; (False, error) on failure.
            Failures are logged but never raise.
        """
        if not self._consent_active():
            return False, "Cloud training disabled. Enable in settings to proceed."
        if not record.consent_flags.get("image_labeling_consent"):
            return False, "Consent required for image upload."

        try:
            if de_id_path := self._deidentify_image(record.image_path):
                record.de_identified_image_path = de_id_path

            from src.training.staging_db import get_staging_db
            get_staging_db().insert_image_label(record)
            logger.info("Captured labeled image: %s", record.image_path)
            return True, "Image labeled and staged for training."
        except Exception as exc:
            logger.exception("Failed to capture image label: %s", exc)
            return False, f"Failed to save image: {exc}"

    @staticmethod
    def _deidentify_image(image_path: str) -> Optional[str]:
        """Copy image to training directory (stub for PHI scrubbing).

        Currently a pass-through; in production would apply image-level
        de-identification (e.g. DICOM tag removal, pixel-level redaction).

        Returns: path to de-identified copy, or None on failure.
        """
        try:
            from pathlib import Path
            from src.features.file_manager import imaging_training_dir
            dest = imaging_training_dir() / Path(image_path).name
            dest.write_bytes(Path(image_path).read_bytes())
            return str(dest)
        except Exception as exc:
            logger.warning("Image de-identification failed (non-critical): %s", exc)
            return None


# ---------------------------------------------------------------------------
# Module-level accessor
# ---------------------------------------------------------------------------

def get_correction_collector() -> CorrectionCollector:
    """Return the process-wide CorrectionCollector singleton."""
    return CorrectionCollector()
