"""Runs ON Kaggle (GPU). Fine-tunes Parakeet TDT 0.6b v3, exports the ONNX folder the app's engine loads, at the precision in parakeet.json.

Reads /kaggle/input/rd-train-data/{manifest.jsonl,audio.tar} (our synthetic radiology clips), the Kaggle
medical-speech dataset, five Hugging Face sets and two recorded clinical-talk sets (PriMock57 from GitHub,
Fareez from figshare); settings come from train/parakeet.json.
Writes /kaggle/working/model_onnx/ (onnx_asr.load_model("nemo-conformer-tdt", path, quantization=cfg["quantization"]))
+ train_log.json, which also holds stock vs tuned error rates on four held-out sets the model never trained on,
+ train_texts.txt, every training line, so the local eval sets can be leak-checked against it,
+ decodes/<set>.json, the tuned model's raw text for every clip of the local eval sets (eval_decode.py).

This file is the run, step by step; the parts it calls ship beside the training data (run_train.py
KERNEL_FILES): sources.py reads audio, talk.py reads the clinical-talk sets, long_clips.py cuts long clips,
augment.py adds noisy copies,
export_onnx.py writes the onnx-asr folder, eval_decode.py decodes the local eval sets, and text_rules.py
holds the text rules the local scripts share.

Training text keeps capitals and punctuation, because Parakeet writes them: a set without them would teach
the model to stop. Rows with spelled-out numbers are dropped for the same reason.
"""
import csv
import gc
import hashlib
import json
import os
import random
import subprocess
import sys
import tarfile
import time
from pathlib import Path

cfg = {}  # kaggle_io.run_kernel writes train/parakeet.json in here at push time

# Fail loudly on a stale dataset version (manifest and the shipped modules), before any paid GPU time: Kaggle can
# start this job before it has finished processing a fresh upload, and then
# mounts the previous one.
DATA = next(p.parent for p in Path("/kaggle/input").rglob("manifest.jsonl")
            if (p.parent / "audio.tar").exists() or (p.parent / "audio").is_dir())
_want = cfg.get("data_sha256")
_got = hashlib.sha256(b"".join((DATA / n).read_bytes()
                              for n in ["manifest.jsonl", *cfg.get("kernel_files", [])])).hexdigest()
if _want and _got != _want:
    raise SystemExit(f"stale training data: manifest + modules sha256 {_got[:12]} != uploaded {_want[:12]}")

subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "onnxruntime"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *cfg["pip"]], check=True)

sys.path.insert(0, str(DATA))  # the kernel's own modules ship flat beside manifest.jsonl
import soundfile as sf  # noqa: E402
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less memory lost to gaps
import torch  # noqa: E402
import onnx_asr  # noqa: E402
import onnxruntime as ort  # noqa: E402

from augment import add_noisy_copies  # noqa: E402
from export_onnx import copy_precision, export  # noqa: E402
from long_clips import Cutter  # noqa: E402
from sources import SR, column, decode, eval_set, hf_files, parallel, parquet_rows, read_clip  # noqa: E402
from talk import Slips, fareez, primock  # noqa: E402
from text_rules import leak_report, normalise, parrot_text, sentence_case, sentences, usable  # noqa: E402

# Scoring on the GPU: on the CPU it took 54 of v7's 213 minutes. torch has loaded the CUDA libraries
# onnxruntime needs; preload_dlls (newer builds) points it at them.
getattr(ort, "preload_dlls", lambda: None)()
# The GPU is shared with torch (the model being trained sits on it): onnxruntime's default doubles its
# memory pool each time it grows, which ran the T4 out; growing by what is asked for does not.
PROVIDERS = [(p, {"arena_extend_strategy": "kSameAsRequested"}) if p == "CUDAExecutionProvider" else p
             for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
print("onnxruntime providers", PROVIDERS, flush=True)

OUT = Path("/kaggle/working")
TMP = Path("/tmp/rd")
CLIPS = TMP / "clips"
CLIPS.mkdir(parents=True, exist_ok=True)
rng = random.Random(cfg["seed"])
log = {"settings": cfg, "sources": {}, "eval": {}}
started = time.time()


def save_log(**kw) -> None:
    """Written after every phase, so a run that dies still says how far it got."""
    log.update(kw, elapsed_min=round((time.time() - started) / 60, 1))
    (OUT / "train_log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(kw)[:2000], flush=True)


# ---------- data ----------
train_rows: list = []


def add_train(source: str, items) -> None:
    """items yields (audio array, text). Keeps those that fit the length and text rules, up to the cap."""
    kept = seen = 0
    cap = cfg["caps"][source]
    dropped = {"too_short": 0, "too_long": 0, "text": 0}
    for a, text in items:
        seen += 1
        dur = len(a) / SR
        why = ("too_short" if dur < cfg["min_duration_sec"] else "too_long" if dur > cfg["max_duration_sec"]
               else None if usable(text) else "text")
        if why:
            dropped[why] += 1
            continue
        path = CLIPS / f"{source}_{kept:05d}.flac"
        sf.write(path, a, SR)
        train_rows.append({"audio_filepath": str(path), "duration": round(dur, 3), "text": text, "source": source})
        kept += 1
        if kept >= cap:
            break
    log["sources"][source] = {"seen": seen, "kept": kept, "dropped": dropped}
    print(f"{source}: kept {kept} of {seen} looked at; dropped {dropped}", flush=True)
    if f"{source}_split" in log["sources"]:
        print(f"{source} long clips cut:", log["sources"][f"{source}_split"], flush=True)


cutter = Cutter(lambda: onnx_asr.load_model(cfg["stock_onnx_model"], providers=PROVIDERS).with_timestamps(),
                cfg["max_duration_sec"], cfg["cut_lead_sec"], cfg["align_window_sec"])


def split_long(source: str, items, known=None):
    return cutter.split(items, log["sources"].setdefault(f"{source}_split", {}), known)


def hf(repo: str, prefix: str, n: int) -> list:
    return hf_files(repo, prefix, n, TMP / "hf")


def rows_of(files: list, *text_cols: str):
    return parquet_rows(files, text_cols, rng)


def held_out(rows) -> list:
    return eval_set(rows, cfg["min_duration_sec"], cfg["eval_max_duration_sec"], cfg["eval_clips_per_set"])


# 1. Our synthetic radiology dictation (rd-train-data).
AUDIO = DATA / "audio"
if not AUDIO.is_dir():
    AUDIO = TMP / "audio"
    tarfile.open(DATA / "audio.tar").extractall(AUDIO)
synth = [json.loads(ln) for ln in (DATA / "manifest.jsonl").read_text().splitlines() if ln.strip()]
add_train("radiology_synth", ((sf.read(AUDIO / r["audio"], dtype="float32")[0], r["text"]) for r in synth))

# 2. Kaggle medical speech: real people reading symptom phrases. train+test to train, validate held out.
csv_path = next(Path("/kaggle/input").rglob("overview-of-recordings.csv"))
wavs = {p.name: p for p in Path("/kaggle/input").rglob("*.wav")}
with open(csv_path, newline="", encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))
fcol, pcol = column(list(rows[0]), "file_name"), column(list(rows[0]), "phrase")
rows = [r for r in rows if r[fcol] in wavs]
rng.shuffle(rows)
held = [r for r in rows if wavs[r[fcol]].parent.name == "validate"]
train_med = [r for r in rows if wavs[r[fcol]].parent.name != "validate"]
add_train("kaggle_medical", zip(parallel(read_clip, [str(wavs[r[fcol]]) for r in train_med]),
                                (r[pcol].strip() for r in train_med)))
evals = {"kaggle_medical_validate": held_out((wavs[r[fcol]].read_bytes(), r[pcol]) for r in held)}

# 3. EdAcc: real accented English conversation (validation to train, test held out).
add_train("edacc", ((decode(b), sentence_case(t)) for b, t in
                    rows_of(hf("edinburghcstr/edacc", "data/validation-", cfg["edacc_train_shards"]), "text")))
evals["edacc_test"] = held_out((b, sentence_case(t)) for b, t in rows_of(
    hf("edinburghcstr/edacc", "data/test-", cfg["edacc_eval_shards"]), "text"))

# 4. Hani89: synthetic medical speech, long clips, cut at sentence ends.
add_train("hani89", split_long("hani89", ((decode(b), t.strip()) for b, t in rows_of(
    hf("Hani89/Synthetic-Medical-Speech-Dataset", "data/train-", cfg["hani89_shards"]), "text"))))

# 5. Eka Care: real Indian-accented medical speech. Eval only, never trained on.
evals["ekacare_en"] = held_out(rows_of(
    hf("ekacare/eka-medical-asr-evaluation-dataset", "en/", cfg["ekacare_eval_shards"]),
    "text", "transcription", "sentence"))

# 6. LibriSpeech train-clean-100: everyday read English, so the model keeps normal speech (v3 lost a little).
#    The local libri check uses test-clean: different speakers, never trained on.
add_train("librispeech", ((decode(b), sentence_case(t)) for b, t in rows_of(
    hf("openslr/librispeech_asr", "clean/train.100/", cfg["libri_train_shards"]), "text")))

# 7. PARROT: radiology reports written by radiologists (MSK included), read by Kokoro TTS voices and cut at
#    pauses into pieces under 30 s, then at sentence ends to fit max_duration_sec. Train split to train.
#    Licence CC-BY-NC-SA 4.0: a model trained on it is for non-commercial use only.
add_train("parrot", split_long("parrot", ((decode(b), parrot_text(t)) for b, t in rows_of(
    hf(cfg["parrot_repo"], "data/train-", cfg["parrot_train_shards"]), "transcription"))))

# 8. Real recorded clinical talk with hand-made transcripts (talk.py): PriMock57 GP consultations (CC BY 4.0)
#    and Fareez OSCE interviews (CC0), whole consultations held out. A training line with a transcriber's slip
#    ("athsma", talk.Slips) is dropped. Held-out sets skip that check (it asks the stock model: no bias to it).
import symspellpy  # noqa: E402

KNOWN = {ln.split()[0] for ln in (Path(symspellpy.__file__).parent / "frequency_dictionary_en_82_765.txt")
         .read_text(encoding="utf-8").splitlines() if ln.strip()}
KNOWN |= {w for ln in (DATA / "radiology_lexicon.txt").read_text(encoding="utf-8").splitlines()
          if not ln.startswith("#") for w in normalise(ln).split()}
SLIPS = Slips(KNOWN)
TALK = TMP / "talk"
every = cfg["held_every"]
add_train("primock", split_long("primock", primock(TALK, every["primock"], held=False), SLIPS))
add_train("fareez", split_long("fareez", fareez(cfg["fareez_url"], TALK, every["fareez"], held=False), SLIPS))


def held_talk(items) -> list:
    """Lines of 3+ words, mixed across every held-out consultation (in order they would all come from the first)."""
    rows = [(a, t) for a, t in items if usable(t)]
    rng.shuffle(rows)
    return held_out(rows)


evals["primock_held"] = held_talk(primock(TALK, every["primock"], held=True))
evals["fareez_held"] = held_talk(split_long("fareez_held", fareez(cfg["fareez_url"], TALK, every["fareez"], held=True)))
cutter.free()  # every long clip is cut: give the aligner's GPU memory back before scoring and training

# PARROT's test split is a random cut of the same reports' pieces, not a split by report, and reports share
# stock sentences ("No free fluid."). So a test piece is only scored if none of its sentences is a training line's.
train_sentences = set().union(*(sentences(r["text"]) for r in train_rows))
parrot_seen = {"looked_at": 0, "shares_a_training_sentence": 0}


def parrot_test_rows():
    for b, t in rows_of(hf(cfg["parrot_repo"], "data/test-", 1), "transcription"):
        t = parrot_text(t)
        parrot_seen["looked_at"] += 1
        if sentences(t) & train_sentences:
            parrot_seen["shares_a_training_sentence"] += 1
            continue
        yield b, t


evals["parrot_test"] = held_out(parrot_test_rows())
log["sources"]["parrot_test"] = parrot_seen

# 9. Noisy copies of a share of every source.
log["sources"].update(add_noisy_copies(train_rows, cfg, rng, TMP, CLIPS))

# Leak check: how much of each held-out set's text the model has seen in training.
# Kaggle "validate" overlaps by design (same phrase list), so it is not used.
texts = sorted({r["text"] for r in train_rows})
(OUT / "train_texts.txt").write_text("\n".join(texts) + "\n", encoding="utf-8")
log["leak"] = leak_report(texts, {name: [t for _, t in items] for name, items in evals.items()})
rng.shuffle(train_rows)
manifest = TMP / "train_manifest.jsonl"
manifest.write_text("\n".join(json.dumps({k: r[k] for k in ("audio_filepath", "duration", "text")})
                              for r in train_rows) + "\n")
save_log(train_clips=len(train_rows), train_hours=round(sum(r["duration"] for r in train_rows) / 3600, 2),
         eval_clips={k: len(v) for k, v in evals.items()})

# ---------- scoring ----------
from nemo.collections.asr.metrics.wer import word_error_rate  # noqa: E402


def score(asr) -> dict:
    res, n = {}, cfg["score_batch"]
    for name, items in evals.items():
        hyps = []
        for i in range(0, len(items), n):
            hyps += asr.recognize([a for a, _ in items[i:i + n]], sample_rate=SR)
        res[name] = round(100 * word_error_rate([normalise(h) for h in hyps], [normalise(t) for _, t in items]), 2)
    return res


log["eval"]["stock"] = score(onnx_asr.load_model(cfg["stock_onnx_model"], quantization=cfg["quantization"], providers=PROVIDERS))
save_log(eval=log["eval"])

# ---------- training ----------
import lightning.pytorch as pl  # noqa: E402
import nemo.collections.asr as nemo_asr  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

pl.seed_everything(cfg["seed"])
model = nemo_asr.models.ASRModel.from_pretrained(cfg["base_model"])
# The early encoder layers hear sounds, the late ones map them to words: keep the early ones as they are.
model.encoder.pre_encode.requires_grad_(False)
for layer in model.encoder.layers[: cfg["freeze_encoder_layers"]]:
    layer.requires_grad_(False)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable params {trainable / 1e6:.0f}M; fused joint {model.joint.fuse_loss_wer}", flush=True)
# BatchNorm keeps a running average of the audio it hears, updated on every training-mode pass even when
# nothing is learning. Batches of two clips make that average noisy, so it stays as NVIDIA trained it.
norms = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]


def hold_norm_stats() -> None:
    for m in norms:
        m.eval()


if cfg.get("check_export"):
    # The untouched model through our export: if this already differs from stock, the recipe is at fault.
    # A second copy from the download cache: the model being trained cannot be copied
    # (NeMo holds an object proxy deepcopy refuses) and must not be exported.
    untouched = nemo_asr.models.ASRModel.from_pretrained(cfg["base_model"], map_location="cpu")
    export(TMP / "untouched", untouched, cfg["onnx_repo"], TMP / "stock")
    for q in (None, "int8"):
        log["eval"][f"untouched_{q or 'fp32'}"] = score(onnx_asr.load_model("nemo-conformer-tdt", str(TMP / "untouched"),
                                                                             quantization=q, providers=PROVIDERS))
    save_log(eval=log["eval"])


def data_cfg(bs: int, path: Path, shuffle: bool) -> DictConfig:
    return DictConfig({"manifest_filepath": str(path), "sample_rate": SR, "batch_size": bs, "shuffle": shuffle,
                       "num_workers": 2, "pin_memory": True, "max_duration": cfg["max_duration_sec"],
                       "min_duration": cfg["min_duration_sec"], "trim_silence": False,
                       "use_start_end_token": False, "use_lhotse": False, "is_tarred": False})


def two_steps(batch, opt) -> None:
    """Forward, backward and an optimiser step, twice, on one batch."""
    sig, sig_len, tr, tr_len = batch
    for _ in range(2):
        with torch.autocast("cuda", dtype=torch.float16):
            enc, enc_len = model.forward(input_signal=sig, input_signal_length=sig_len)
            dec, tgt_len, _ = model.decoder(targets=tr, target_length=tr_len)
            if model.joint.fuse_loss_wer:
                loss = model.joint(encoder_outputs=enc, decoder_outputs=dec, encoder_lengths=enc_len,
                                   transcripts=tr, transcript_lengths=tr_len, compute_wer=False)[0]
            else:
                loss = model.loss(log_probs=model.joint(encoder_outputs=enc, decoder_outputs=dec), targets=tr,
                                  input_lengths=enc_len, target_lengths=tgt_len)
        loss.backward()
        opt.step()
        del enc, dec, loss


def fits(bs: int) -> bool:
    """Two full training steps (forward, backward, optimiser state) on `bs` copies of the longest clip.

    Two, because AdamW only creates its state (twice the trainable weights) at the first step: one step
    peaks before that state exists, so it passed batch 4 that then ran out of memory at step 2 (v4 smoke)."""
    # A batch is padded to its longest audio AND its longest text, so pair the two worst clips.
    worst = [max(train_rows, key=lambda r: r["duration"]), max(train_rows, key=lambda r: len(r["text"]))]
    probe = TMP / "probe.jsonl"
    probe.write_text("\n".join(json.dumps({k: worst[i % 2][k] for k in ("audio_filepath", "duration", "text")})
                               for i in range(bs)))
    model.cuda().train()
    hold_norm_stats()
    batch = [t.cuda() for t in next(iter(model._setup_dataloader_from_config(data_cfg(bs, probe, False))))]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.0)
    # The fused joint clears its loss reduction while it runs and restores it after; an
    # out-of-memory mid-run skips the restore, and training then gets one loss per clip.
    reduction = model.joint.loss.reduction if model.joint.fuse_loss_wer else None
    try:
        two_steps(batch, opt)
        ok = True
    except torch.cuda.OutOfMemoryError:
        ok = False
    finally:
        if model.joint.fuse_loss_wer:
            model.joint.loss.reduction = reduction
    peak = torch.cuda.max_memory_allocated() / 2**30
    del opt, batch
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"probe batch {bs}: {'fits' if ok else 'out of memory'} (peak {peak:.1f} GB)", flush=True)
    return ok


bs = cfg["batch_size"]
while not fits(bs):
    if bs == 1:
        raise SystemExit("even one clip at max_duration_sec does not fit; lower it")
    bs //= 2
accumulate = max(1, cfg["effective_batch"] // bs)


class Progress(pl.Callback):
    last = 0  # with gradient accumulation several batches share one step; print each step once

    def on_train_batch_start(self, *_):
        hold_norm_stats()  # Lightning switches every layer back to training mode each epoch

    def on_train_batch_end(self, trainer, *_):
        step = trainer.global_step
        if step % 50 == 0 and step != self.last:
            self.last = step
            loss = float(trainer.callback_metrics.get("train_loss", float("nan")))
            log.setdefault("loss_every_50_steps", []).append(round(loss, 4))
            # Peak since the last report: a climb from one report to the next is a leak, a jump is one bad batch.
            peak = torch.cuda.max_memory_allocated() / 2**30
            torch.cuda.reset_peak_memory_stats()
            log.setdefault("peak_gb_every_50_steps", []).append(round(peak, 1))
            print(f"step {step}  loss {loss:.4f}  peak {peak:.1f} GB  {(time.time() - started) / 60:.0f} min", flush=True)


class Average(pl.Callback):
    """Running sum of the trainable weights every `average_every` steps over the last `average_last_steps`
    (on the CPU, in fp32). The mean of several late points usually sits nearer the middle of the low ground
    than any one step does (checkpoint averaging). The last step is always one of the points."""

    def __init__(self, every: int, last: int, max_steps: int):
        self.every, self.start, self.n, self.sum, self.taken = every, max_steps - last, 0, None, set()

    def take(self, pl_module, step: int) -> None:
        if step in self.taken:
            return
        self.taken.add(step)
        weights = {k: p.detach().float().cpu() for k, p in pl_module.named_parameters() if p.requires_grad}
        self.sum = weights if self.sum is None else {k: self.sum[k] + v for k, v in weights.items()}
        self.n += 1

    def on_train_batch_end(self, trainer, pl_module, *_):
        step = trainer.global_step
        if step > self.start and step % self.every == 0:
            self.take(pl_module, step)

    def on_train_end(self, trainer, pl_module):
        self.take(pl_module, trainer.global_step)

    def apply(self, model) -> int:
        with torch.no_grad():
            for k, p in model.named_parameters():
                if k in self.sum:
                    p.copy_((self.sum[k] / self.n).to(p.device, p.dtype))
        return self.n


hours = cfg["max_train_hours"]
average = Average(cfg["average_every"], cfg["average_last_steps"], cfg["max_steps"])
trainer = pl.Trainer(devices=1, accelerator="gpu", precision=cfg["precision"], max_steps=cfg["max_steps"],
                     max_time=f"00:{int(hours):02d}:{int(hours % 1 * 60):02d}:00", accumulate_grad_batches=accumulate,
                     gradient_clip_val=1.0, limit_val_batches=0, num_sanity_val_steps=0, logger=False,
                     enable_checkpointing=False, enable_progress_bar=False, log_every_n_steps=10,
                     callbacks=[Progress(), average])
model.set_trainer(trainer)
model.setup_training_data(data_cfg(bs, manifest, True))
model.setup_optimization(DictConfig({
    "name": "adamw", "lr": cfg["learning_rate"], "betas": [0.9, 0.98], "weight_decay": cfg["weight_decay"],
    "sched": {"name": "CosineAnnealing", "warmup_steps": cfg["warmup_steps"], "min_lr": cfg["min_learning_rate"]}}))
stopped = "finished"
try:
    trainer.fit(model)
except torch.cuda.OutOfMemoryError as e:
    stopped = "out of memory part-way; exported what it had learned so far"
    print(e, flush=True)  # how much it tried to take and how much was already held
    log["out_of_memory"] = str(e)[:500]
save_log(batch_size=bs, accumulate=accumulate, steps_done=trainer.global_step, training=stopped,
         averaged_points=average.apply(model), averaged_steps=sorted(average.taken))

# ---------- export ----------
export(TMP / "tuned", model, cfg["onnx_repo"], TMP / "stock")
# Export moved the model to the CPU, but AdamW's two running averages per weight stay on the GPU: v9's first
# full run ran out of GPU memory scoring the tuned model because of them. Drop them before onnxruntime starts.
for opt in trainer.optimizers:
    opt.state.clear()
gc.collect()
torch.cuda.empty_cache()
save_log(gpu_gb_held_after_training=round(torch.cuda.memory_allocated() / 2**30, 2))
dest = OUT / cfg["output"]
save_log(exported=copy_precision(TMP / "tuned", dest, cfg["quantization"]))

log["eval"]["tuned"] = score(onnx_asr.load_model("nemo-conformer-tdt", str(dest), quantization=cfg["quantization"], providers=PROVIDERS))
if cfg.get("check_export"):
    # Full precision too: int8 rounds each weight, so a change far too small to learn anything still tips
    # some weights into the next step. Comparing fp32 to fp32 tells training apart from rounding.
    log["eval"]["tuned_fp32"] = score(onnx_asr.load_model("nemo-conformer-tdt", str(TMP / "tuned"), providers=PROVIDERS))
    # How far training moved anything: weights by the learning rate, the held BatchNorm averages not at all.
    names, after, change = {n for n, _ in model.named_parameters()}, model.state_dict(), {"weights": 0.0, "buffers": 0.0}
    for k, v in untouched.state_dict().items():
        kind = "weights" if k in names else "buffers"
        change[kind] = max(change[kind], float((after[k].cpu().double() - v.double()).abs().max()) if v.numel() else 0.0)
    log["max_change"] = change
save_log(eval=log["eval"], max_change=log.get("max_change"))

if cfg.get("score_eval_sets"):
    from eval_decode import decode_eval_sets
    decode_eval_sets(TMP / "tuned", cfg, PROVIDERS, TMP, OUT, save_log)
