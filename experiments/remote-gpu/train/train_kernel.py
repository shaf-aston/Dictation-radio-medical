"""Runs ON Kaggle (GPU). Fine-tunes Whisper, exports a CTranslate2 folder.

Reads /kaggle/input/rd-train-data/{manifest.jsonl,config.json,*.flac}
Writes /kaggle/working/model_ct2/ (what faster-whisper loads) + train_log.json
"""
import json, os, random, subprocess, sys, tarfile, time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ctranslate2"], check=True)

import numpy as np, soundfile as sf, torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Kaggle's mount path differs between runtimes; find the data by its manifest.
DATA = next(Path("/kaggle/input").rglob("manifest.jsonl")).parent
print("data at", DATA, sorted(p.name for p in DATA.iterdir())[:5], flush=True)
OUT = Path("/kaggle/working")
# Kaggle unpacks the uploaded audio.tar into audio/; extract by hand only if it did not.
AUDIO = DATA / "audio"
if not AUDIO.is_dir():
    AUDIO = Path("/tmp/audio")
    tarfile.open(DATA / "audio.tar").extractall(AUDIO)
cfg = json.loads((DATA / "config.json").read_text())
rows = [json.loads(l) for l in (DATA / "manifest.jsonl").read_text().splitlines() if l.strip()]
random.Random(0).shuffle(rows)
n_val = max(1, int(len(rows) * cfg["val_fraction"]))
val, train = rows[:n_val], rows[n_val:]
print(f"train {len(train)}  val {len(val)}", flush=True)

proc = WhisperProcessor.from_pretrained(cfg["base_model"])
model = WhisperForConditionalGeneration.from_pretrained(cfg["base_model"]).cuda()
model.config.forced_decoder_ids = None
if cfg["freeze_encoder"]:
    for p in model.model.encoder.parameters():
        p.requires_grad = False


def batch_of(items):
    audio = []
    for r in items:
        a, sr = sf.read(AUDIO / r["audio"], dtype="float32")
        audio.append(a[: int(cfg["max_clip_sec"] * sr)])
    feats = proc.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features
    ids = [proc.tokenizer(r["text"]).input_ids for r in items]
    sot = model.config.decoder_start_token_id
    ids = [i[1:] if i and i[0] == sot else i for i in ids]  # model prepends SOT itself
    width = max(map(len, ids))
    labels = torch.full((len(ids), width), -100, dtype=torch.long)
    for k, i in enumerate(ids):
        labels[k, : len(i)] = torch.tensor(i)
    return feats.cuda(), labels.cuda()


def val_loss():
    model.eval()
    tot = 0.0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(val), cfg["batch_size"]):
            x, y = batch_of(val[i : i + cfg["batch_size"]])
            tot += model(input_features=x, labels=y).loss.item() * len(y)
    model.train()
    return tot / len(val)


opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["learning_rate"])
steps_total = cfg["epochs"] * ((len(train) + cfg["batch_size"] - 1) // cfg["batch_size"])
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / steps_total))
scaler = torch.amp.GradScaler()
log = {"config": cfg, "train": len(train), "val": len(val), "val_loss_start": val_loss(), "steps": []}
print("val loss before:", log["val_loss_start"], flush=True)

model.train()
t0, step = time.time(), 0
for ep in range(cfg["epochs"]):
    random.Random(ep).shuffle(train)
    for i in range(0, len(train), cfg["batch_size"]):
        x, y = batch_of(train[i : i + cfg["batch_size"]])
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(input_features=x, labels=y).loss
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step(); step += 1
        if step % 20 == 0:
            print(f"ep {ep} step {step}/{steps_total} loss {loss.item():.4f}", flush=True)
            log["steps"].append([step, loss.item()])
    log[f"val_loss_ep{ep}"] = val_loss()
    print(f"epoch {ep} val loss {log[f'val_loss_ep{ep}']:.4f}", flush=True)
log["train_sec"] = time.time() - t0

hf = OUT / "hf_model"
model.save_pretrained(hf); proc.save_pretrained(hf)
subprocess.run(["ct2-transformers-converter", "--model", str(hf), "--output_dir", str(OUT / "model_ct2"),
                "--quantization", cfg["quantization"], "--copy_files", "tokenizer.json"], check=True)
# newer transformers no longer writes preprocessor_config.json: take the base model's
from huggingface_hub import hf_hub_download
import shutil
# same for tokenizer.json: the new format can't be read by faster-whisper's older tokenizers
for f in ("preprocessor_config.json", "tokenizer.json"):
    shutil.copy(hf_hub_download(cfg["base_model"], f), OUT / "model_ct2" / f)
subprocess.run(["rm", "-rf", str(hf)])
(OUT / "train_log.json").write_text(json.dumps(log, indent=1))
print("EXPORT OK", os.listdir(OUT / "model_ct2"))
