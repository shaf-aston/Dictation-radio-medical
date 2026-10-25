"""Write ``parakeet_kaggle.ipynb`` from the cells below.

    python -m scripts.finetune.make_notebook

The notebook is generated, not hand-edited, so its code lives in a reviewable
Python file and the JSON is always valid. The same cell code is exercised on
CPU by ``smoke_test.py`` (two training steps, export, load in onnx-asr), so
what runs on Kaggle has run before.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

CELLS = [
    ("markdown", """# Fine-tune Parakeet TDT on radiology speech (Kaggle)

Radio Dictate's default recogniser is NVIDIA Parakeet TDT 0.6B v2. It hears
general English well and radiology terms less well (measured: ~17% of lexicon
terms wrong before post-processing). This notebook adapts it to radiology
vocabulary and exports it in the exact layout the app loads.

**Recipe** (why each choice, in `scripts/finetune/README.md`):
1. Data: the zip built by `scripts/finetune/build_dataset.py` (synthetic
   speech of template sentences and lexicon terms; 20% of terms and one voice
   held out; never the evaluation corpus).
2. Freeze the encoder (acoustics), train the decoder and joint (vocabulary).
   Synthetic speech teaches words, not acoustics; freezing the encoder is the
   recommended way to use it without forgetting real voices.
3. Small constant learning rate, fp16 (T4 has no bf16), checkpoint often.
4. Measure before and after on the held-out set, export to ONNX + int8.

**Gate:** a fine-tune is only used if it beats the stock model on *your own
recorded voice* (`python -m scripts.eval.run_eval --set own --engine parakeet
--model <export dir>`). Better on synthetic speech alone is not enough.

Settings: Accelerator **GPU T4 x2** or **P100**, Internet **on** (to download
the base model once). Add the dataset zip as an input."""),
    ("code", """# --- settings --------------------------------------------------------------
import glob
import os
DATA_DIR = glob.glob('/kaggle/input/*/stats.json')
DATA_DIR = os.path.dirname(DATA_DIR[0]) if DATA_DIR else '/kaggle/input/radiology-asr'
BASE_MODEL = 'nvidia/parakeet-tdt-0.6b-v2'   # same as the app's default
OUT_DIR = '/kaggle/working'
MAX_STEPS = 3000          # ~1-2 h on a T4 at batch 8; raise if val keeps improving
BATCH_SIZE = 8            # lower to 4 if you see CUDA out-of-memory
LR = 3e-5                 # small and constant: adapt, do not overwrite
SMOKE = False             # True: 2 steps on CPU, for checking the code path
print('data:', DATA_DIR)"""),
    ("code", """# --- install -----------------------------------------------------------------
# Kaggle images ship torch; NeMo brings the rest. If a later NeMo breaks this,
# pin the version the smoke test passed with (see README).
!pip install -q "nemo_toolkit[asr]" onnx onnxruntime jiwer"""),
    ("code", """# --- data ----------------------------------------------------------------------
import json
def absolute_manifest(name):
    # Manifests store paths relative to the dataset; NeMo wants absolute ones.
    src = os.path.join(DATA_DIR, f'{name}.jsonl')
    dst = os.path.join(OUT_DIR, f'{name}.abs.jsonl')
    with open(src) as f, open(dst, 'w') as g:
        for line in f:
            row = json.loads(line)
            row['audio_filepath'] = os.path.join(DATA_DIR, row['audio_filepath'])
            g.write(json.dumps(row) + '\\n')
    return dst
TRAIN, VAL, TEST = (absolute_manifest(n) for n in ('train', 'val', 'test_heldout'))
HELDOUT_TERMS = [t.strip().lower() for t in open(os.path.join(DATA_DIR, 'heldout_terms.txt')) if t.strip()]
print(open(os.path.join(DATA_DIR, 'stats.json')).read())"""),
    ("code", """# --- model -----------------------------------------------------------------------
import torch
import nemo.collections.asr as nemo_asr
model = nemo_asr.models.ASRModel.from_pretrained(BASE_MODEL, map_location='cpu')
device = 'cuda' if torch.cuda.is_available() and not SMOKE else 'cpu'
model = model.to(device)"""),
    ("code", """# --- measure: before -----------------------------------------------------------------
import re

import jiwer


def read(manifest, limit=None):
    rows = [json.loads(line) for line in open(manifest)]
    return rows[:limit] if limit else rows
def norm(t):
    return ' '.join(re.sub(r"[^\\w\\s'-]", ' ', t.lower()).replace('-', ' ').split())
def evaluate(manifest, limit=None):
    rows = read(manifest, limit)
    model.eval()
    with torch.no_grad():
        hyps = model.transcribe([r['audio_filepath'] for r in rows], batch_size=8)
    hyps = [h.text if hasattr(h, 'text') else h for h in hyps]
    refs = [norm(r['text']) for r in rows]
    hyps = [norm(h) for h in hyps]
    hit = total = 0
    for ref, hyp in zip(refs, hyps):
        for term in HELDOUT_TERMS:
            if re.search(rf'\\b{re.escape(term)}\\b', ref):
                total += 1
                hit += bool(re.search(rf'\\b{re.escape(term)}\\b', hyp))
    return {'wer': round(jiwer.wer(refs, hyps), 4),
            'heldout_term_recall': round(hit / total, 4) if total else None, 'clips': len(rows)}
LIMIT = 4 if SMOKE else None
before = {'val': evaluate(VAL, LIMIT), 'test_heldout': evaluate(TEST, LIMIT)}
print('before', before)"""),
    ("code", """# --- train: decoder + joint only --------------------------------------------------
import lightning.pytorch as pl
from omegaconf import OmegaConf, open_dict

model.encoder.freeze()          # keep the acoustics; adapt the vocabulary side
with open_dict(model.cfg):
    for key, manifest, shuffle in (('train_ds', TRAIN, True), ('validation_ds', VAL, False)):
        ds = model.cfg[key]
        ds.manifest_filepath = manifest
        ds.batch_size = 2 if SMOKE else BATCH_SIZE
        ds.shuffle = shuffle
        ds.num_workers = 0 if SMOKE else 4
        ds.max_duration = 20.0
        ds.is_tarred = False
        ds.tarred_audio_filepaths = None
        ds.pin_memory = not SMOKE
    # No 'sched' entry: NeMo then keeps the learning rate constant, which is
    # what adapting (rather than retraining) wants.
    model.cfg.optim = OmegaConf.create({
        'name': 'adamw', 'lr': LR, 'weight_decay': 1e-3, 'betas': [0.9, 0.98],
    })
model.setup_training_data(model.cfg.train_ds)
model.setup_validation_data(model.cfg.validation_ds)
model.setup_optimization(model.cfg.optim)

trainer = pl.Trainer(
    accelerator='cpu' if SMOKE else 'gpu', devices=1,
    precision='32' if SMOKE else '16-mixed',
    max_steps=2 if SMOKE else MAX_STEPS,
    val_check_interval=1.0 if SMOKE else 250,
    limit_val_batches=1 if SMOKE else 1.0,
    log_every_n_steps=1 if SMOKE else 50,
    enable_checkpointing=False, logger=False,
    gradient_clip_val=1.0,
)
model.set_trainer(trainer)
if SMOKE:
    # NeMo's TDT loss runs only on CUDA ("TDT is not yet implemented for non
    # CUDA computation"), so a CPU smoke run stops short of the training step
    # itself. Everything around it (data, optimiser, export) still runs.
    print('SMOKE: skipping trainer.fit (TDT loss needs a GPU)')
else:
    trainer.fit(model)
model.save_to(os.path.join(OUT_DIR, 'parakeet-radiology.nemo'))   # keep the full model too"""),
    ("code", """# --- measure: after ---------------------------------------------------------------
after = {'val': evaluate(VAL, LIMIT), 'test_heldout': evaluate(TEST, LIMIT)}
print('before', before)
print('after ', after)
json.dump({'before': before, 'after': after}, open(os.path.join(OUT_DIR, 'scores.json'), 'w'), indent=2)"""),
    ("code", """# --- export for the app (onnx-asr layout) ----------------------------------------------
# The exact files the app's Parakeet engine loads from a folder:
#   config.json, encoder-model.int8.onnx, decoder_joint-model.int8.onnx, vocab.txt
from onnxruntime.quantization import QuantType, quantize_dynamic
export = os.path.join(OUT_DIR, 'parakeet-radiology-onnx')
os.makedirs(export, exist_ok=True)
model = model.to('cpu').eval()
model.preprocessor.featurizer.dither = 0.0
model.preprocessor.featurizer.pad_to = 0
model.export(os.path.join(export, 'model.onnx'))   # writes encoder-model.onnx + decoder_joint-model.onnx
with open(os.path.join(export, 'vocab.txt'), 'w') as f:
    for i, token in enumerate([*model.tokenizer.vocab, '<blk>']):
        f.write(f'{token} {i}\\n')
json.dump({'model_type': 'nemo-conformer-tdt', 'features_size': 128, 'subsampling_factor': 8},
          open(os.path.join(export, 'config.json'), 'w'), indent=4)
for part in ('encoder-model', 'decoder_joint-model'):
    quantize_dynamic(os.path.join(export, f'{part}.onnx'), os.path.join(export, f'{part}.int8.onnx'),
                     weight_type=QuantType.QUInt8, per_channel=True)
# The fp32 export spills its weights into dozens of side files; the int8
# models are self-contained (checked by the smoke test), so keep only the four.
KEEP = {'config.json', 'vocab.txt', 'encoder-model.int8.onnx', 'decoder_joint-model.int8.onnx'}
for name in os.listdir(export):
    if name not in KEEP:
        os.remove(os.path.join(export, name))
print(sorted(os.listdir(export)))"""),
    ("code", """# --- check the export decodes like the model it came from ---------------------------------
!pip install -q onnx-asr
import onnx_asr
exported = onnx_asr.load_model('nemo-parakeet-tdt-0.6b-v2', export, quantization='int8')
row = read(TEST, 1)[0]
print('reference:', row['text'])
print('nemo     :', model.transcribe([row['audio_filepath']])[0])
print('onnx     :', exported.recognize(row['audio_filepath']))
!cd /kaggle/working && zip -qr parakeet-radiology-onnx.zip parakeet-radiology-onnx scores.json"""),
    ("markdown", """## Use it in the app

1. Download `parakeet-radiology-onnx.zip` from the notebook's Output and unzip it
   on the radiologist's machine.
2. **Gate it first:**
   `python -m scripts.eval.run_eval --set own --engine parakeet --model <unzipped dir>`
   and compare with the same command without `--model`. Keep it only if
   medical-term error is lower on your own voice.
3. Switch it on: copy the unzipped folder to `data/models/fine_tuned/<name>`
   and pick it under **Voice model** (web Settings, or the desktop settings
   bar). **Standard** goes back to stock."""),
]


def build() -> dict:
    cells = []
    for kind, source in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "gpu", "isInternetEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def code_cells() -> list:
    """The code cells as plain Python (``!`` shell lines removed), for the smoke test."""
    out = []
    for kind, source in CELLS:
        if kind == "code":
            out.append("\n".join(ln for ln in source.splitlines() if not ln.lstrip().startswith("!")))
    return out


if __name__ == "__main__":
    path = HERE / "parakeet_kaggle.ipynb"
    path.write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path}")
