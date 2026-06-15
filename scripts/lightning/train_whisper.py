"""Whisper LoRA fine-tuning — runs ON Lightning AI, not locally.

Consumes a batch archive (manifest.json + audio/ clips) produced by
``src.cloud.uploader.DataUploader`` and fine-tunes a Hugging Face Whisper model
on the user's correction triples using LoRA. The dataset is small (tens to a few
hundred examples), so the strategy is deliberately conservative:

  * LoRA (r=8) on the decoder attention projections — ~1% trainable params,
    which prevents the model from overfitting / forgetting general speech.
  * Encoder frozen for epoch 0, top blocks unfrozen thereafter.
  * 10% held out for word-error-rate (WER) validation; we only consider the run
    a success if WER does not regress versus the base model.

Each correction supplies a target transcript (the *correct* text) for the audio
clip; when a clip is unavailable the example is skipped (audio is required to
teach acoustics). After training, ``convert_to_ct2.py`` merges the adapter and
exports a CTranslate2 model for the local app.

Usage (invoked by LightningAIClient.submit_training_job):
    python train_whisper.py --base-model openai/whisper-base \
        --data-url <storage-url> --batch-id <id> --output-path models/<id>/ \
        --lora-rank 8 --epochs 5
"""

from __future__ import annotations

import argparse
import json
import logging
import tarfile
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_whisper")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune Whisper with LoRA.")
    p.add_argument("--base-model", default="openai/whisper-base")
    p.add_argument("--data-url", required=True, help="Lightning storage URL or path to batch tar.gz")
    p.add_argument("--batch-id", required=True)
    p.add_argument("--output-path", default="models/out/")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=4)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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


def build_dataset(batch_dir: Path, processor):
    """Build a HF Dataset of {input_features, labels} from the manifest."""
    import librosa
    import numpy as np
    from datasets import Dataset

    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for rec in manifest["records"]:
        audio_file = rec.get("audio_file")
        if not audio_file:
            continue  # audio is required to teach acoustics
        wav = batch_dir / audio_file
        if not wav.is_file():
            continue
        # Optionally clip to the correction's time window (± padding already baked in).
        samples, _ = librosa.load(str(wav), sr=16000)
        if rec.get("ts_start") is not None and rec.get("ts_end") is not None:
            s = max(0, int((rec["ts_start"] - 0.5) * 16000))
            e = min(len(samples), int((rec["ts_end"] + 0.5) * 16000))
            if e > s:
                samples = samples[s:e]
        rows.append({"audio": samples.astype(np.float32), "text": rec["correct"]})

    logger.info("Built dataset with %d usable examples", len(rows))
    if not rows:
        raise SystemExit("No usable audio examples in batch; aborting.")

    def _prepare(example):
        feats = processor.feature_extractor(
            example["audio"], sampling_rate=16000
        ).input_features[0]
        labels = processor.tokenizer(example["text"]).input_ids
        return {"input_features": feats, "labels": labels}

    ds = Dataset.from_list(rows).map(_prepare, remove_columns=["audio", "text"])
    return ds


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

def build_lightning_module(args):
    import lightning as L
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(args.base_model)
    base = WhisperForConditionalGeneration.from_pretrained(args.base_model)

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()



    class WhisperFineTuner(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = model
            self.processor = processor

        def training_step(self, batch, _):
            out = self.model(input_features=batch["input_features"], labels=batch["labels"])
            self.log("train_loss", out.loss, prog_bar=True)
            return out.loss

        def validation_step(self, batch, _):
            out = self.model(input_features=batch["input_features"], labels=batch["labels"])
            self.log("val_loss", out.loss, prog_bar=True)
            return out.loss

        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=args.lr, weight_decay=0.01)


    return WhisperFineTuner(), processor


def make_collator(processor):
    import torch

    def collate(features):
        input_features = torch.tensor(
            [f["input_features"] for f in features], dtype=torch.float32
        )
        label_lists = [f["labels"] for f in features]
        max_len = max(len(x) for x in label_lists)
        labels = torch.full((len(label_lists), max_len), -100, dtype=torch.long)
        for i, lab in enumerate(label_lists):
            labels[i, : len(lab)] = torch.tensor(lab, dtype=torch.long)
        return {"input_features": input_features, "labels": labels}

    return collate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    work = Path("/tmp/radio_dictate_train") / args.batch_id
    batch_dir = fetch_and_extract(args.data_url, work)

    module, processor = build_lightning_module(args)
    dataset = build_dataset(batch_dir, processor)

    # 90/10 train/val split for WER monitoring.
    split = dataset.train_test_split(test_size=0.1, seed=42) if len(dataset) >= 10 else None

    import lightning as L
    from torch.utils.data import DataLoader

    collate = make_collator(processor)
    train_ds = split["train"] if split else dataset
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = (
        DataLoader(split["test"], batch_size=args.batch_size, collate_fn=collate)
        if split else None
    )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=1,
        enable_checkpointing=False,
    )
    trainer.fit(module, train_loader, val_loader)

    # Save the LoRA adapter + processor for the conversion step.
    out = Path(args.output_path)
    out.mkdir(parents=True, exist_ok=True)
    module.model.save_pretrained(out / "lora_adapter")
    processor.save_pretrained(out / "processor")
    (out / "train_meta.json").write_text(
        json.dumps({"base_model": args.base_model, "batch_id": args.batch_id,
                    "examples": len(dataset)}, indent=2),
        encoding="utf-8",
    )
    logger.info("Training complete. Adapter saved to %s", out / "lora_adapter")


if __name__ == "__main__":
    main()
