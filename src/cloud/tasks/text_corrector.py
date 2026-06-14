"""Text-correction fine-tuning task.

Trains a small seq2seq model (default ``t5-small``) to map a raw transcript span
to its corrected form, learning the radiologist's recurring fixes that the rule
-based post-processing pipeline misses. Unlike the voice task this needs no
audio — only the (wrong → correct) text pairs already staged for voice training,
so its archive is a single ``manifest.json``.

Archive layout (consumed by ``scripts/lightning/train_text_corrector.py``):
    manifest.json   — list of {wrong, correct} pairs
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
from src.training.schemas import TASK_TEXT_CORRECTOR, CorrectionRecord

logger = logging.getLogger(__name__)


class TextCorrectorTask:
    """Bundles wrong→correct text pairs and submits a seq2seq job."""

    task_type = TASK_TEXT_CORRECTOR

    def build_archive(
        self, batch_id: str, records: List[CorrectionRecord], base_model: str
    ) -> Path:
        """Write a tar.gz containing only manifest.json (text pairs, no audio)."""
        manifest = {
            "batch_id": batch_id,
            "base_model": base_model,
            "task_type": self.task_type,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pairs": [
                {"wrong": r.wrong_text, "correct": r.correct_text}
                for r in records
                if r.wrong_text and r.correct_text and r.wrong_text != r.correct_text
            ],
        }
        tmp = Path(tempfile.gettempdir()) / f"{batch_id}.tar.gz"
        with tarfile.open(tmp, "w:gz") as tar:
            data = json.dumps(manifest, indent=2).encode("utf-8")
            info = tarfile.TarInfo("manifest.json")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        logger.info("Built text-corrector batch %s (%d pairs)",
                    tmp.name, len(manifest["pairs"]))
        return tmp

    def job_spec(
        self, batch_id: str, data_url: str, base_model: str,
        epochs: int = 8, **_,
    ) -> JobSpec:
        # Text-only seq2seq is light; a single mid-tier GPU is plenty.
        return JobSpec(
            name=f"radio-dictate-text-{batch_id}",
            entrypoint="scripts/lightning/train_text_corrector.py",
            compute={"type": "gpu", "name": "T4"},
            args={
                "base-model": base_model or "t5-small",
                "batch-id": batch_id,
                "data-url": data_url,
                "epochs": epochs,
                "output-path": f"models/{batch_id}/",
            },
        )
