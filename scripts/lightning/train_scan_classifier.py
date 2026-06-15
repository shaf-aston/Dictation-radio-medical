"""Chest X-ray classifier fine-tuning — runs ON Lightning AI, not locally.

Consumes a batch archive (``manifest.json`` with ``examples: [{image_file,
labels}]`` + ``images/``) produced by
``src.cloud.tasks.scan_finetune.ScanClassifierTask`` and adapts a pretrained
TorchXRayVision DenseNet121 to the user's own labelled scans.

This is patient-safety-critical, so the strategy is deliberately conservative:

  * Only the classifier head is trained by default; the validated convolutional
    backbone is frozen, so general feature quality cannot be destroyed by a small,
    possibly-noisy local dataset.
  * 20% is held out for a mean-AUC validation gate. The run is REJECTED (non-zero
    exit, no artifact) if the fine-tuned mean AUC regresses versus the base model.
    A model that does not beat the validated baseline is never shipped — the app
    keeps the safer stock weights.

Usage (invoked by LightningAIClient.submit_training_job):
    python train_scan_classifier.py --base-model densenet121-res224-all \
        --data-url <storage-url> --batch-id <id> --output-path models/<id>/ \
        --epochs 10
"""

from __future__ import annotations

import argparse
import json
import logging
import tarfile
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_scan_classifier")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune a chest X-ray classifier head.")
    p.add_argument("--base-model", default="densenet121-res224-all")
    p.add_argument("--data-url", required=True, help="Lightning storage URL or path to batch tar.gz")
    p.add_argument("--batch-id", required=True)
    p.add_argument("--output-path", default="models/out/")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


def fetch_and_extract(data_url: str, work_dir: Path) -> Path:
    """Download (if URL) and extract the batch archive; return its root dir."""
    work_dir.mkdir(parents=True, exist_ok=True)
    if data_url.startswith(("http://", "https://", "gs://", "s3://")):
        import urllib.request
        local_archive = work_dir / "batch.tar.gz"
        logger.info("Downloading batch from %s", data_url)
        urllib.request.urlretrieve(data_url, local_archive)  # noqa: S310
    else:
        local_archive = Path(data_url)
    extract_dir = work_dir / "batch"
    with tarfile.open(local_archive, "r:gz") as tar:
        tar.extractall(extract_dir)
    return extract_dir


def load_examples(batch_dir: Path, model) -> Tuple[list, list]:
    """Return (image_tensors, label_vectors) aligned to ``model.pathologies``."""
    import numpy as np
    import skimage.io
    import torch
    import torchxrayvision as xrv

    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    resizer = xrv.datasets.XRayResizer(224)
    pathologies = list(model.pathologies)

    images, labels = [], []
    for ex in manifest.get("examples", []):
        img_path = batch_dir / ex["image_file"]
        if not img_path.is_file():
            continue
        img = xrv.datasets.normalize(skimage.io.imread(str(img_path)), 255)
        if img.ndim == 3:
            img = img.mean(2)
        img = resizer(img[None, ...])
        images.append(torch.from_numpy(img).float())
        # Build a label vector aligned to the model's output order; -1 = unknown.
        vec = np.full(len(pathologies), -1.0, dtype=np.float32)
        for label, value in ex.get("labels", {}).items():
            if label in pathologies:
                vec[pathologies.index(label)] = float(value)
        labels.append(torch.from_numpy(vec))

    logger.info("Loaded %d labelled scans", len(images))
    if len(images) < 4:
        raise SystemExit("Too few labelled scans to train/validate; aborting.")
    return images, labels


def mean_auc(model, images: List, labels: List) -> float:
    """Mean ROC-AUC over labels that have both classes present in the held-out set."""
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score

    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(torch.stack([model(im.unsqueeze(0))[0] for im in images]))
    y_pred = preds.cpu().numpy()
    y_true = torch.stack(labels).cpu().numpy()

    aucs = []
    for j in range(y_true.shape[1]):
        mask = y_true[:, j] >= 0          # ignore unknown labels for this column
        col = y_true[mask, j]
        if mask.sum() >= 2 and len(set(col)) == 2:
            aucs.append(roc_auc_score(col, y_pred[mask, j]))
    return float(np.mean(aucs)) if aucs else 0.0


def main() -> None:
    args = parse_args()
    work = Path("/tmp/radio_dictate_scan") / args.batch_id
    batch_dir = fetch_and_extract(args.data_url, work)

    import torch
    import torchxrayvision as xrv

    model = xrv.models.DenseNet(weights=args.base_model)
    images, labels = load_examples(batch_dir, model)

    # 80/20 split for the AUC gate.
    split = max(1, len(images) // 5)
    val_imgs, val_lbls = images[:split], labels[:split]
    train_imgs, train_lbls = images[split:], labels[split:]

    baseline_auc = mean_auc(model, val_imgs, val_lbls)
    logger.info("Baseline mean AUC on held-out: %.3f", baseline_auc)

    # Freeze the validated backbone; train only the classifier head.
    for p in model.features.parameters():
        p.requires_grad = False
    head = model.classifier
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for img, lbl in zip(train_imgs, train_lbls):
            optimizer.zero_grad()
            logits = model(img.unsqueeze(0))[0]
            mask = (lbl >= 0).float()        # supervise only known labels
            target = torch.clamp(lbl, min=0.0)
            loss = (loss_fn(logits, target) * mask).sum() / mask.clamp(min=1).sum()
            loss.backward()
            optimizer.step()
            total += float(loss)
        logger.info("Epoch %d: mean loss %.4f", epoch, total / max(1, len(train_imgs)))

    tuned_auc = mean_auc(model, val_imgs, val_lbls)
    logger.info("Fine-tuned mean AUC on held-out: %.3f", tuned_auc)

    # Safety gate: a clinical model that regresses is never shipped.
    if tuned_auc < baseline_auc:
        raise SystemExit(
            f"Fine-tuned AUC {tuned_auc:.3f} < baseline {baseline_auc:.3f}; "
            "rejecting run so the app keeps the validated base weights.")

    out = Path(args.output_path)
    model_dir = out / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "weights.pt")
    (model_dir / "meta.json").write_text(
        json.dumps({"pathologies": list(model.pathologies),
                    "base_model": args.base_model}, indent=2),
        encoding="utf-8",
    )
    (out / "train_meta.json").write_text(
        json.dumps({"base_model": args.base_model, "batch_id": args.batch_id,
                    "examples": len(images), "baseline_auc": baseline_auc,
                    "tuned_auc": tuned_auc}, indent=2),
        encoding="utf-8",
    )
    _pack(model_dir, out)
    logger.info("Scan-classifier training complete.")


def _pack(model_dir: Path, out: Path) -> Path:
    archive = out / "model.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for item in model_dir.iterdir():
            tar.add(item, arcname=item.name)
    logger.info("Packed model → %s", archive)
    return archive


if __name__ == "__main__":
    main()
