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

import io
import json
import tarfile
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Protocol, runtime_checkable

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


def base_manifest(batch_id: str, task_type: str, base_model: str) -> dict:
    """Return the four fields every task manifest starts with."""
    return {
        "batch_id": batch_id,
        "base_model": base_model,
        "task_type": task_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def base_job_args(
    batch_id: str, data_url: str, base_model: str, epochs: int
) -> dict:
    """Return the five CLI args common to every training job."""
    return {
        "base-model": base_model,
        "batch-id": batch_id,
        "data-url": data_url,
        "epochs": epochs,
        "output-path": f"models/{batch_id}/",
    }


def write_manifest_archive(
    batch_id: str, manifest: dict, members: Optional[Dict[str, str]] = None
) -> Path:
    """Write *manifest* as ``manifest.json`` plus any *members* into a tar.gz.

    *members* maps a source file path to its archive name (e.g.
    ``{"/abs/clip.wav": "audio/clip.wav"}``). Every task's ``build_archive``
    bundles a manifest the same way; only the manifest contents and which extra
    files ride along differ. Returns the path to a temp ``<batch_id>.tar.gz``
    the framework uploads then deletes.
    """
    tmp = Path(tempfile.gettempdir()) / f"{batch_id}.tar.gz"
    with tarfile.open(tmp, "w:gz") as tar:
        data = json.dumps(manifest, indent=2).encode("utf-8")
        info = tarfile.TarInfo("manifest.json")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
        for src_path, arcname in (members or {}).items():
            tar.add(src_path, arcname=arcname)
    return tmp
