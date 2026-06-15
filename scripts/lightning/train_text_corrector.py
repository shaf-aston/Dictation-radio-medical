"""Text-correction fine-tuning — runs ON Lightning AI, not locally.

Consumes a batch archive (``manifest.json`` with ``pairs: [{wrong, correct}]``)
produced by ``src.cloud.tasks.text_corrector.TextCorrectorTask`` and fine-tunes a
small seq2seq model (default ``t5-small``) to map a raw transcript span to its
corrected form. The dataset is small (the user's own recurring fixes), so the
strategy is conservative:

  * Small model + a few epochs; 10% held out for an exact-match validation gate.
  * The run is only considered a success if validation exact-match does not
    regress versus the untuned base — the text analogue of the voice WER gate.

After training, the model directory is packed into ``model.tar.gz`` for download
by the app's SyncManager. The local report-cleanup path can then load it.

Usage (invoked by LightningAIClient.submit_training_job):
    python train_text_corrector.py --base-model t5-small \
        --data-url <storage-url> --batch-id <id> --output-path models/<id>/ \
        --epochs 8
"""

from __future__ import annotations

import argparse
import json
import logging
import tarfile
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_text_corrector")

_PREFIX = "correct: "   # T5 task prefix


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune a seq2seq text corrector.")
    p.add_argument("--base-model", default="t5-small")
    p.add_argument("--data-url", required=True, help="Lightning storage URL or path to batch tar.gz")
    p.add_argument("--batch-id", required=True)
    p.add_argument("--output-path", default="models/out/")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
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


def load_pairs(batch_dir: Path) -> List[Tuple[str, str]]:
    """Read ``manifest.json`` into a list of (wrong, correct) pairs."""
    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    pairs = [(p["wrong"], p["correct"]) for p in manifest.get("pairs", [])
             if p.get("wrong") and p.get("correct")]
    logger.info("Loaded %d correction pairs", len(pairs))
    if not pairs:
        raise SystemExit("No usable correction pairs in batch; aborting.")
    return pairs


def build_dataset(pairs: List[Tuple[str, str]], tokenizer):
    """Tokenise pairs into a HF Dataset of {input_ids, labels}."""
    from datasets import Dataset

    rows = [{"src": _PREFIX + w, "tgt": c} for w, c in pairs]

    def _prepare(ex):
        model_inputs = tokenizer(ex["src"], truncation=True, max_length=256)
        labels = tokenizer(text_target=ex["tgt"], truncation=True, max_length=256)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return Dataset.from_list(rows).map(_prepare, remove_columns=["src", "tgt"])


def exact_match(model, tokenizer, pairs: List[Tuple[str, str]]) -> float:
    """Fraction of held-out pairs the model corrects exactly."""
    import torch

    if not pairs:
        return 0.0
    hits = 0
    model.eval()
    with torch.no_grad():
        for wrong, correct in pairs:
            ids = tokenizer(_PREFIX + wrong, return_tensors="pt").input_ids
            out = model.generate(ids, max_length=256)
            pred = tokenizer.decode(out[0], skip_special_tokens=True)
            if pred.strip() == correct.strip():
                hits += 1
    return hits / len(pairs)


def main() -> None:
    args = parse_args()
    work = Path("/tmp/radio_dictate_text") / args.batch_id
    batch_dir = fetch_and_extract(args.data_url, work)
    pairs = load_pairs(batch_dir)

    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)

    # 90/10 split so we can gate on exact-match regression.
    split = max(1, len(pairs) // 10)
    val_pairs, train_pairs = pairs[:split], pairs[split:]
    baseline_em = exact_match(base, tokenizer, val_pairs)
    logger.info("Baseline exact-match on held-out: %.3f", baseline_em)

    train_ds = build_dataset(train_pairs, tokenizer)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    out = Path(args.output_path)
    targs = Seq2SeqTrainingArguments(
        output_dir=str(out / "trainer"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
    )
    trainer = Seq2SeqTrainer(
        model=model, args=targs, train_dataset=train_ds, data_collator=collator,
    )
    trainer.train()

    tuned_em = exact_match(model, tokenizer, val_pairs)
    logger.info("Fine-tuned exact-match on held-out: %.3f", tuned_em)

    # Safety gate: never ship a model that regresses versus the base.
    if tuned_em < baseline_em:
        raise SystemExit(
            f"Fine-tuned exact-match {tuned_em:.3f} < baseline {baseline_em:.3f}; "
            "rejecting run so the app keeps the better base behaviour.")

    model_dir = out / "model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    (out / "train_meta.json").write_text(
        json.dumps({"base_model": args.base_model, "batch_id": args.batch_id,
                    "pairs": len(pairs), "baseline_em": baseline_em,
                    "tuned_em": tuned_em}, indent=2),
        encoding="utf-8",
    )
    _pack(model_dir, out)
    logger.info("Text-corrector training complete.")


def _pack(model_dir: Path, out: Path) -> Path:
    archive = out / "model.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for item in model_dir.iterdir():
            tar.add(item, arcname=item.name)
    logger.info("Packed model → %s", archive)
    return archive


if __name__ == "__main__":
    main()
