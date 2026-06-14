"""The contract every cloud-training task implements.

A :class:`TrainingTask` is the seam between the generic framework and a specific
model type. It supplies the three things that genuinely differ per model:

  1. how to bundle pending records into a batch archive (``build_archive``);
  2. how to describe the Lightning job that trains on that archive (``job_spec``);
  3. the ``task_type`` tag its produced models carry in the registry.

Everything else — uploading the archive, polling the job, downloading and
extracting the artifact, registering the model version — lives in the framework
and is identical across tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Protocol, runtime_checkable

from src.training.schemas import CorrectionRecord


@dataclass
class JobSpec:
    """A Lightning AI job description produced by a task.

    Attributes:
        name: Human-readable job name (shows up in the Lightning dashboard).
        entrypoint: Repo-relative training script the job runs.
        compute: Lightning compute selector, e.g. ``{"type": "gpu", "name": "A10G"}``.
        args: CLI args passed to the entrypoint, as a flat ``{flag: value}`` map.
    """

    name: str
    entrypoint: str
    compute: dict
    args: dict = field(default_factory=dict)


@runtime_checkable
class TrainingTask(Protocol):
    """Per-model-type plug-in for the cloud training framework."""

    #: Stable tag (see ``src.training.schemas.TASK_*``) used to key the registry,
    #: batches, and the artifact destination directory.
    task_type: str

    def build_archive(
        self, batch_id: str, records: List[CorrectionRecord], base_model: str
    ) -> Path:
        """Bundle *records* into a ``.tar.gz`` the training script can consume.

        Returns the path to a temp archive; the framework uploads then deletes it.
        """
        ...

    def job_spec(
        self, batch_id: str, data_url: str, base_model: str, **hyperparams
    ) -> JobSpec:
        """Describe the Lightning job that trains on the uploaded *data_url*."""
        ...
