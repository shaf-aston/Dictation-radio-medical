"""Merge a trained LoRA adapter into Whisper and export to CTranslate2.

Runs ON Lightning AI after ``train_whisper.py``. The local app uses
faster-whisper (CTranslate2), which cannot load a Hugging Face / LoRA model
directly — so this script:

  1. Loads the base Whisper model and applies the saved LoRA adapter.
  2. Merges the adapter weights into the base (``merge_and_unload``).
  3. Runs ``ct2-transformers-converter`` to produce an int8 CT2 model.
  4. Packs the CT2 directory into ``model.tar.gz`` for download by the app's
     SyncManager, which extracts it into ``data/models/fine_tuned/<version>/``.

Usage:
    python convert_to_ct2.py --train-output models/<id>/ --output models/<id>/ct2/
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import tarfile
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("convert_to_ct2")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge LoRA + export Whisper to CTranslate2.")
    p.add_argument("--train-output", required=True,
                   help="Directory produced by train_whisper.py (contains lora_adapter/).")
    p.add_argument("--output", required=True, help="Destination directory for the CT2 model.")
    p.add_argument("--quantization", default="int8")
    return p.parse_args()


def merge_adapter(train_output: Path, merged_dir: Path) -> str:
    """Merge the LoRA adapter into the base model; return the base model name."""
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    meta = json.loads((train_output / "train_meta.json").read_text(encoding="utf-8"))
    base_model = meta["base_model"]

    logger.info("Loading base model %s", base_model)
    base = WhisperForConditionalGeneration.from_pretrained(base_model)
    logger.info("Applying LoRA adapter")
    merged = PeftModel.from_pretrained(base, str(train_output / "lora_adapter"))
    merged = merged.merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(merged_dir)
    WhisperProcessor.from_pretrained(str(train_output / "processor")).save_pretrained(merged_dir)
    logger.info("Merged model written to %s", merged_dir)
    return base_model


def convert_ct2(merged_dir: Path, output: Path, quantization: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ct2-transformers-converter",
        "--model", str(merged_dir),
        "--output_dir", str(output),
        "--quantization", quantization,
        "--force",
    ]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def pack(output: Path) -> Path:
    archive = output.parent / "model.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for item in output.iterdir():
            tar.add(item, arcname=item.name)
    logger.info("Packed CT2 model → %s", archive)
    return archive


def main() -> None:
    args = parse_args()
    train_output = Path(args.train_output)
    output = Path(args.output)

    with tempfile.TemporaryDirectory() as tmp:
        merged_dir = Path(tmp) / "merged"
        merge_adapter(train_output, merged_dir)
        convert_ct2(merged_dir, output, args.quantization)

    pack(output)
    logger.info("Conversion complete.")


if __name__ == "__main__":
    main()
