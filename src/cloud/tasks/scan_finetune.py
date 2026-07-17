"""Scan-classifier fine-tuning task (chest X-ray).

Fine-tunes the classifier head of a TorchXRayVision DenseNet121 on the user's
own labelled chest X-rays so the local Scan Assistant adapts to their equipment
and population. This is **opt-in and separately consented**: scan images are far
more sensitive than text, so this task only runs when the user has explicitly
enabled scan training and provided labelled data.

Unlike the voice/text tasks, the training examples are images, not correction
triples — so this task bundles a labelled-image directory rather than reading the
correction staging DB. The framework still handles upload/poll/download/register
unchanged.

Archive layout (consumed by ``scripts/lightning/train_scan_classifier.py``):
    manifest.json   — [{image_file, labels: {pathology: 0|1}}]
    images/*.png    — the (locally de-identified) images

Safety gate: the training script holds out a validation split and **rejects the
run** if the fine-tuned model's mean AUC regresses versus the base weights — the
mirror of the voice path's WER gate. A model that does not beat the validated
baseline is never registered.

Specialty focus — chest trauma: rib/clavicle fractures correlate clinically with
pneumothorax and haemothorax (the latter reads as effusion on a plain film), and
plain films are documented to miss a large share of rib fractures. So when a
batch contains labelled positives for :data:`_TRAUMA_FOCUS_LABELS`, this task
oversamples them — the standard, training-script-agnostic way to sharpen a
multi-label fine-tune toward a chosen subset without touching model code or the
abstention thresholds (those stay a separate, per-site calibration lever; see
``data/imaging/thresholds.json``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

from src.cloud.exceptions import CloudError
from src.cloud.tasks.base import JobSpec, base_job_args, base_manifest, write_manifest_archive
from src.imaging.datasets import parse_label_file
from src.imaging.schemas import DEFAULT_SCAN_WEIGHTS
from src.training.schemas import TASK_SCAN_CLASSIFIER, CorrectionRecord

logger = logging.getLogger(__name__)

# Labels this task's fine-tune specialises toward (see module docstring). A
# labelled example that is positive for any of these is repeated
# _TRAUMA_OVERSAMPLE times in the manifest so the fine-tune sees proportionally
# more trauma cases than the stock 18-label baseline does.
_TRAUMA_FOCUS_LABELS = frozenset({"Fracture", "Pneumothorax", "Effusion"})
_TRAUMA_OVERSAMPLE = 3


def _is_trauma_positive(labels: dict) -> bool:
    """True if *labels* marks any chest-trauma-focus pathology positive (1)."""
    return any(labels.get(label) == 1 for label in _TRAUMA_FOCUS_LABELS)


class ScanClassifierTask:
    """Bundles labelled chest X-rays and submits a classifier fine-tune job."""

    task_type = TASK_SCAN_CLASSIFIER

    def build_archive(
        self, batch_id: str, records: List[CorrectionRecord], base_model: str
    ) -> Path:
        """Build the image batch from registered datasets or the legacy training directory.

        Scan examples are not correction triples, so *records* is unused; the
        labelled images come from registered datasets (via DatasetRegistry) or,
        for backward compatibility, from ``imaging_training_dir()``. Each image
        must sit beside a ``<name>.json`` holding ``{"pathology": 0|1, ...}`` labels.

        Datasets are aggregated: if multiple datasets are registered, the manifest
        includes examples from all of them. Trauma-positive examples
        (see :data:`_TRAUMA_FOCUS_LABELS`) are listed :data:`_TRAUMA_OVERSAMPLE`
        times in the manifest — the underlying image is still archived only once.

        Backward compatibility: If no datasets are registered, falls back to
        reading from the legacy ``imaging_training_dir()``.
        """
        from src.features.file_manager import imaging_training_dir
        from src.imaging.datasets import DatasetRegistry

        registry = DatasetRegistry()
        dataset_names = registry.list_datasets()

        # Collect source directories: registered datasets or fallback to legacy dir
        src_dirs: List[Path] = []
        if dataset_names:
            for ds_name in dataset_names:
                try:
                    dataset = registry.get_dataset(ds_name)
                    src_dirs.append(Path(dataset["path"]))
                    logger.info("Including dataset '%s' in batch", ds_name)
                except (KeyError, ValueError) as exc:
                    logger.warning("Skipping dataset '%s': %s", ds_name, exc)
        else:
            # Backward compat: use the legacy training directory
            legacy_dir = imaging_training_dir()
            if list(legacy_dir.glob("*.png")):
                src_dirs.append(legacy_dir)
                logger.info("No registered datasets; falling back to legacy training dir")

        if not src_dirs:
            raise CloudError("No labelled scans available to train on.")

        resolved_model = base_model or DEFAULT_SCAN_WEIGHTS
        manifest = {
            **base_manifest(batch_id, self.task_type, resolved_model),
            "trauma_focus": {
                "labels": sorted(_TRAUMA_FOCUS_LABELS),
                "oversample_factor": _TRAUMA_OVERSAMPLE,
            },
            "examples": [],
        }
        members: dict = {}
        trauma_positives = 0

        # Aggregate from all source directories
        for src_dir in src_dirs:
            for img in sorted(src_dir.glob("*.png")):
                label_file = img.with_suffix(".json")
                if not label_file.is_file():
                    continue  # an image without labels can't supervise training
                try:
                    labels, _attributes = parse_label_file(label_file)
                except (OSError, json.JSONDecodeError, ValueError) as exc:
                    logger.warning("Skipping unreadable scan label %s: %s", label_file, exc)
                    continue
                arcname = f"images/{img.name}"
                members[str(img)] = arcname
                example = {"image_file": arcname, "labels": labels}
                if _is_trauma_positive(labels):
                    trauma_positives += 1
                    manifest["examples"].extend([example] * _TRAUMA_OVERSAMPLE)
                else:
                    manifest["examples"].append(example)

        if not manifest["examples"]:
            raise CloudError("No labelled scans available to train on.")

        tmp = write_manifest_archive(batch_id, manifest, members)
        logger.info(
            "Built scan batch %s: %d images -> %d training examples "
            "(%d trauma-positive, oversampled %dx)",
            tmp.name, len(members), len(manifest["examples"]),
            trauma_positives, _TRAUMA_OVERSAMPLE)
        return tmp

    def job_spec(
        self, batch_id: str, data_url: str, base_model: str,
        epochs: int = 10, **_,
    ) -> JobSpec:
        resolved_model = base_model or DEFAULT_SCAN_WEIGHTS
        return JobSpec(
            name=f"radio-dictate-scan-{batch_id}",
            entrypoint="scripts/lightning/train_scan_classifier.py",
            compute={"type": "gpu", "name": "A10G"},
            args=base_job_args(batch_id, data_url, resolved_model, epochs),
        )
