"""Cloud-training tasks — one module per ML model type.

Each task plugs into :mod:`src.cloud.framework`: it knows how to bundle its own
training data into an archive, how to describe its Lightning job (entrypoint,
compute, args), and what ``task_type`` tag its models carry in the registry. The
framework handles everything generic (upload, poll, download, register).

Available tasks are listed in :data:`TASKS`, keyed by ``task_type``.
"""

from __future__ import annotations

from typing import Dict

from src.cloud.tasks.base import TrainingTask
from src.cloud.tasks.scan_finetune import ScanClassifierTask
from src.cloud.tasks.text_corrector import TextCorrectorTask
from src.cloud.tasks.whisper_voice import WhisperVoiceTask

# Registry of available tasks, keyed by their ``task_type``.
TASKS: Dict[str, TrainingTask] = {
    t.task_type: t
    for t in (WhisperVoiceTask(), TextCorrectorTask(), ScanClassifierTask())
}

__all__ = [
    "TASKS",
    "TrainingTask",
    "WhisperVoiceTask",
    "TextCorrectorTask",
    "ScanClassifierTask",
]
