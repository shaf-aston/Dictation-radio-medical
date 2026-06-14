"""Whisper voice fine-tuning task.

Trains a LoRA adapter over a Hugging Face Whisper model on the user's
(audio, wrong, correct) correction triples, then converts the result to a
CTranslate2 model the local faster-whisper transcriber can load.

Archive layout (consumed by ``scripts/lightning/train_whisper.py``):
    manifest.json   — the correction triples + per-record audio_file pointers
    audio/*.wav     — the de-identified clips (deduplicated; many records share one)
"""

from __future__ import annotations

import io
import json
import logging
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from src.cloud.tasks.base import JobSpec
from src.training.schemas import TASK_WHISPER_VOICE, CorrectionRecord

logger = logging.getLogger(__name__)


class WhisperVoiceTask:
    """Bundles correction triples and submits a Whisper LoRA job."""

    task_type = TASK_WHISPER_VOICE

    def build_archive(
        self, batch_id: str, records: List[CorrectionRecord], base_model: str
    ) -> Path:
        """Write a tar.gz of audio clips + manifest.json to a temp file."""
        manifest = {
            "batch_id": batch_id,
            "base_model": base_model,
            "task_type": self.task_type,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "records": [],
        }
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

        tmp = Path(tempfile.gettempdir()) / f"{batch_id}.tar.gz"
        with tarfile.open(tmp, "w:gz") as tar:
            manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
            info = tarfile.TarInfo("manifest.json")
            info.size = len(manifest_bytes)
            tar.addfile(info, io.BytesIO(manifest_bytes))
            for src_path, arcname in audio_members.items():
                tar.add(src_path, arcname=arcname)
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
            args={
                "base-model": base_model,
                "batch-id": batch_id,
                "data-url": data_url,
                "lora-rank": lora_rank,
                "epochs": epochs,
                "output-path": f"models/{batch_id}/",
            },
        )
