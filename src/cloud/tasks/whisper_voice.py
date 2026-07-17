"""Whisper voice fine-tuning task.

Trains a LoRA adapter over a Hugging Face Whisper model on the user's
(audio, wrong, correct) correction triples, then converts the result to a
CTranslate2 model the local faster-whisper transcriber can load.

Archive layout (consumed by ``scripts/lightning/train_whisper.py``):
    manifest.json   — the correction triples + per-record audio_file pointers
    audio/*.wav     — the de-identified clips (deduplicated; many records share one)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from src.cloud.tasks.base import JobSpec, base_job_args, base_manifest, write_manifest_archive
from src.training.schemas import TASK_WHISPER_VOICE, CorrectionRecord

logger = logging.getLogger(__name__)


class WhisperVoiceTask:
    """Bundles correction triples and submits a Whisper LoRA job."""

    task_type = TASK_WHISPER_VOICE

    def build_archive(
        self, batch_id: str, records: List[CorrectionRecord], base_model: str
    ) -> Path:
        """Write a tar.gz of audio clips + manifest.json to a temp file."""
        manifest = {**base_manifest(batch_id, self.task_type, base_model), "records": []}
        # Deduplicate audio clips — many records share one session clip.
        audio_members: dict = {}
        for rec in records:
            entry = {
                "wrong": rec.wrong_text,
                "correct": rec.correct_text,
                "accent": rec.accent_profile,
                "ts_start": rec.ts_start,
                "ts_end": rec.ts_end,
                "audio_file": None,
            }
            if rec.audio_path and Path(rec.audio_path).is_file():
                arcname = f"audio/{Path(rec.audio_path).name}"
                audio_members[rec.audio_path] = arcname
                entry["audio_file"] = arcname
            manifest["records"].append(entry)

        tmp = write_manifest_archive(batch_id, manifest, audio_members)
        logger.info("Built voice batch %s (%d records, %d clips)",
                    tmp.name, len(records), len(audio_members))
        return tmp

    def job_spec(
        self, batch_id: str, data_url: str, base_model: str,
        lora_rank: int = 8, epochs: int = 5, **_,
    ) -> JobSpec:
        return JobSpec(
            name=f"radio-dictate-voice-{batch_id}",
            entrypoint="scripts/lightning/train_whisper.py",
            compute={"type": "gpu", "name": "A10G"},
            args={**base_job_args(batch_id, data_url, base_model, epochs), "lora-rank": lora_rank},
        )
