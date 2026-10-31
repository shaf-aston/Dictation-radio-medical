"""Runs ON Kaggle (GPU). Fine-tunes Parakeet TDT 0.6b v3, exports the ONNX folder the app's engine loads, at the precision in parakeet.json.

Reads /kaggle/input/rd-train-data/{manifest.jsonl,audio.tar} (our synthetic radiology clips), the Kaggle
medical-speech dataset, and five Hugging Face sets; settings come from train/parakeet.json.
Writes /kaggle/working/model_onnx/ (onnx_asr.load_model("nemo-conformer-tdt", path, quantization=cfg["quantization"]))
+ train_log.json, which also holds stock vs tuned error rates on test sets the model never trained on,
+ train_texts.txt, every training line, so the local test sets can be leak-checked against it.

Training text keeps capitals and punctuation, because Parakeet writes them: a set without them would teach
the model to stop. Rows with spelled-out numbers are dropped for the same reason.
"""
import csv
import hashlib
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

cfg = {}  # kaggle_io.run_kernel writes train/parakeet.json in here at push time

subprocess.run([sys.executable, "-m", "pip", "install", "-q", *cfg["pip"]], check=True)

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import soundfile as sf  # noqa: E402
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less memory lost to gaps
import torch  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

SR = 16000
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


# ---------- audio and text ----------
def decode(b: bytes) -> np.ndarray:
    """Any encoded clip (wav/flac/mp3) -> 16 kHz mono float32."""
    try:
        a, sr = sf.read(io.BytesIO(b), dtype="float32")
    except Exception:
        r = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", str(SR),
                            "pipe:1"], input=b, capture_output=True, check=True)
        return np.frombuffer(r.stdout, np.float32)
    if a.ndim > 1:
        a = a.mean(axis=1)
    return a if sr == SR else resample_poly(a, SR, sr).astype(np.float32)


NUMBER_WORDS = re.compile(r"\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
                          r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
                          r"eighty|ninety|hundred|thousand|million|percent)\b", re.I)
ODD_CHARS = re.compile(r"[^A-Za-z0-9' ,.;:?!()/%-]")  # markup, other scripts, symbols: drop the row


def usable(text: str) -> bool:
    return len(text.split()) >= 3 and not NUMBER_WORDS.search(text) and not ODD_CHARS.search(text)


def sentence_case(t: str) -> str:
    """EdAcc and LibriSpeech are UPPERCASE (EdAcc with <LAUGH>-style tags): strip tags, sentence-case it."""
    t = " ".join(re.sub(r"<[^>]*>", " ", t).split()).lower()
    t = re.sub(r"\bi\b", "I", t)
    return t[:1].upper() + t[1:]


def parrot_text(t: str) -> str:
    """PARROT marks line and paragraph breaks with the words NEWLINE and PARAGRAPH: drop them."""
    return " ".join(re.sub(r"(NEWLINE|PARAGRAPH)", " ", t).split())


def normalise(t: str) -> str:
    """For scoring only: lowercase, no punctuation, so only the words are compared."""
    return " ".join(re.sub(r"[^a-z0-9' ]", " ", t.lower()).split())


def sentences(t: str) -> set:
    """The sentences of four words or more, normalised (a decimal point does not end one)."""
    return {s for s in (normalise(x) for x in re.split(r"(?<!\d)[.;:?!](?!\d)", t)) if len(s.split()) >= 4}


def runs(t: str, n: int = 4) -> set:
    """Every run of `n` words in a row, normalised."""
    w = normalise(t).split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def column(names: list, *wanted: str) -> str:
    for w in wanted:
        if w in names:
            return w
    raise SystemExit(f"none of {wanted} in columns {names}")


# ---------- data ----------
train_rows: list = []


def add_train(source: str, items) -> None:
    """items yields (audio array, text). Keeps those that fit the length and text rules, up to the cap."""
    kept = seen = 0
    cap = cfg["caps"][source]
    for a, text in items:
        seen += 1
        dur = len(a) / SR
        if not (cfg["min_duration_sec"] <= dur <= cfg["max_duration_sec"]) or not usable(text):
            continue
        path = CLIPS / f"{source}_{kept:05d}.flac"
        sf.write(path, a, SR)
        train_rows.append({"audio_filepath": str(path), "duration": round(dur, 3), "text": text, "source": source})
        kept += 1
        if kept >= cap:
            break
    log["sources"][source] = {"seen": seen, "kept": kept}
    print(f"{source}: kept {kept} of {seen} looked at", flush=True)


def hf_files(repo: str, prefix: str, n: int) -> list:
    """`n` parquet shards spread evenly across the repo's `prefix` shards (EdAcc groups shards by speaker)."""
    names = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                   if f.startswith(prefix) and f.endswith(".parquet"))
    if not names:
        raise SystemExit(f"no {prefix}*.parquet in {repo}")
    pick = names[:: max(1, len(names) // n)][:n]
    return [hf_hub_download(repo, f, repo_type="dataset", local_dir=TMP / "hf") for f in pick]


def parquet_rows(files: list, text_cols: tuple, audio_col: str = "audio"):
    """Yields (audio bytes, text) from parquet shards, shuffled within each shard."""
    for f in files:
        pf = pq.ParquetFile(f)
        tcol = column(pf.schema_arrow.names, *text_cols)
        rows = pf.read(columns=[tcol, audio_col]).to_pylist()
        rng.shuffle(rows)
        for r in rows:
            a = r[audio_col]
            yield (a["bytes"] if isinstance(a, dict) else a), r[tcol]


def eval_set(rows) -> list:
    out = []
    for b, text in rows:
        a = decode(b)
        if normalise(text) and cfg["min_duration_sec"] <= len(a) / SR <= 30:
            out.append((a, text))
        if len(out) >= cfg["eval_clips_per_set"]:
            break
    return out


# 1. Our synthetic radiology dictation (rd-train-data).
DATA = next(Path("/kaggle/input").rglob("manifest.jsonl")).parent
# Fail loudly on a stale dataset version: Kaggle can start this job before it
# has finished processing a fresh upload, and then mounts the previous one.
_want = cfg.get("data_sha256")
_got = hashlib.sha256((DATA / "manifest.jsonl").read_bytes()).hexdigest()
if _want and _got != _want:
    raise SystemExit(f"stale training data: manifest sha256 {_got[:12]} != uploaded {_want[:12]}")
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
add_train("kaggle_medical", ((decode(wavs[r[fcol]].read_bytes()), r[pcol].strip())
                             for r in rows if wavs[r[fcol]].parent.name != "validate"))
evals = {"kaggle_medical_validate": eval_set((wavs[r[fcol]].read_bytes(), r[pcol]) for r in held)}

# 3. EdAcc: real accented English conversation (validation to train, test held out).
add_train("edacc", ((decode(b), sentence_case(t)) for b, t in
                    parquet_rows(hf_files("edinburghcstr/edacc", "data/validation-", cfg["edacc_train_shards"]),
                                 ("text",))))
evals["edacc_test"] = eval_set((b, sentence_case(t)) for b, t in parquet_rows(
    hf_files("edinburghcstr/edacc", "data/test-", cfg["edacc_eval_shards"]), ("text",)))

# 4. Hani89: synthetic medical speech, long clips, so only the short ones fit.
add_train("hani89", ((decode(b), t.strip()) for b, t in parquet_rows(
    hf_files("Hani89/Synthetic-Medical-Speech-Dataset", "data/train-", cfg["hani89_shards"]), ("text",))))

# 5. Eka Care: real Indian-accented medical speech. Test only, never trained on.
evals["ekacare_en"] = eval_set(parquet_rows(
    hf_files("ekacare/eka-medical-asr-evaluation-dataset", "en/", cfg["ekacare_eval_shards"]),
    ("text", "transcription", "sentence")))

# 6. LibriSpeech train-clean-100: everyday read English, so the model keeps normal speech (v3 lost a little).
#    The local libri check uses test-clean: different speakers, never trained on.
add_train("librispeech", ((decode(b), sentence_case(t)) for b, t in parquet_rows(
    hf_files("openslr/librispeech_asr", "clean/train.100/", cfg["libri_train_shards"]), ("text",))))

# 7. PARROT: radiology reports written by radiologists (MSK included), read by Kokoro TTS voices and cut at
#    pauses into pieces under 30 s (only those up to max_duration_sec are kept). Train split to train.
#    Licence CC-BY-NC-SA 4.0: a model trained on it is for non-commercial use only.
PARROT = "ysdede/parrot-radiology-asr-en"
add_train("parrot", ((decode(b), parrot_text(t)) for b, t in parquet_rows(
    hf_files(PARROT, "data/train-", cfg["parrot_train_shards"]), ("transcription",))))

# PARROT's test split is a random cut of the same reports' pieces, not a split by report, and reports share
# stock sentences ("No free fluid."). So a test piece is only scored if none of its sentences is a training line's.
train_sentences = set().union(*(sentences(r["text"]) for r in train_rows))
parrot_seen = {"looked_at": 0, "shares_a_training_sentence": 0}


def parrot_test_rows():
    for b, t in parquet_rows(hf_files(PARROT, "data/test-", 1), ("transcription",)):
        t = parrot_text(t)
        parrot_seen["looked_at"] += 1
        if sentences(t) & train_sentences:
            parrot_seen["shares_a_training_sentence"] += 1
            continue
        yield b, t


evals["parrot_test"] = eval_set(parrot_test_rows())
log["sources"]["parrot_test"] = parrot_seen


def real_noises_and_rooms() -> tuple:
    """OpenSLR 28 (Apache 2.0): recorded noises, and real room echoes (impulse responses); channels are averaged by decode()."""
    z = TMP / "rirs_noises.zip"
    subprocess.run(["curl", "-sSfL", "-o", str(z), cfg["rir_noise_url"]], check=True)
    subprocess.run(["unzip", "-q", "-o", str(z), "-d", str(TMP)], check=True)
    files = sorted((TMP / "RIRS_NOISES").rglob("*.wav"))
    noises = [f for f in files if "pointsource_noises" in str(f) or ("real_rirs" in str(f) and "noise" in f.name)]
    rooms = [f for f in files if "real_rirs" in str(f) and "rir" in f.name and "noise" not in f.name]
    log["sources"]["openslr28"] = {"noises": len(noises), "rooms": len(rooms)}
    return noises, rooms


# 8. Noisy copies: babble (another clip), recorded noise, or steady hiss, at a random loudness gap; some
#    also in a real room's echo. Not pink noise: the local noise test uses pink, so training on it would flatter it.
def add_noisy_copies() -> None:
    from scipy.signal import fftconvolve
    np_rng = np.random.default_rng(cfg["seed"])
    lo, hi = cfg["noise_snr_db"]
    noises, rooms = real_noises_and_rooms()
    picks = rng.sample(train_rows, int(len(train_rows) * cfg["noisy_copy_fraction"]))
    for i, r in enumerate(picks):
        a = sf.read(r["audio_filepath"], dtype="float32")[0]
        if rng.random() < cfg["reverb_fraction"]:
            room = decode(rng.choice(rooms).read_bytes())
            a = fftconvolve(a, room / (np.max(np.abs(room)) + 1e-9))[: len(a)].astype(np.float32)
        kind = rng.random()
        if kind < 1 / 3:
            noise = np.resize(sf.read(rng.choice(train_rows)["audio_filepath"], dtype="float32")[0], len(a))
        elif kind < 2 / 3:
            noise = np.resize(decode(rng.choice(noises).read_bytes()), len(a))
        else:
            noise = np_rng.standard_normal(len(a)).astype(np.float32)
        gap = 10 ** (rng.uniform(lo, hi) / 20)
        rms = lambda x: float(np.sqrt(np.mean(x ** 2))) + 1e-9  # noqa: E731
        mixed = np.clip(a + noise * (rms(a) / rms(noise) / gap), -1, 1)
        path = CLIPS / f"noisy_{i:05d}.flac"
        sf.write(path, mixed, SR)
        train_rows.append({**r, "audio_filepath": str(path), "source": r["source"] + "+noise"})
    log["sources"]["noisy_copies"] = len(picks)


add_noisy_copies()


# Leak check: how much of each test set's text the model has seen in training. Whole phrases should be 0;
# four-word runs show near-misses. Kaggle "validate" overlaps by design (same phrase list), so it is not used.
def leak_check() -> None:
    texts = sorted({r["text"] for r in train_rows})
    (OUT / "train_texts.txt").write_text("\n".join(texts) + "\n", encoding="utf-8")
    joined = "| " + " | ".join(normalise(t) for t in texts) + " |"  # spaces: match whole words only
    train_runs = set().union(*(runs(t) for t in texts))
    log["leak"] = {}
    for name, items in evals.items():
        phrases = {normalise(t) for _, t in items} - {""}
        test_runs = set().union(*(runs(t) for _, t in items))
        found = sorted(p for p in phrases if f" {p} " in joined)
        log["leak"][name] = {"phrases": len(phrases), "phrases_in_training": len(found), "found": found[:20],
                             "four_word_runs": len(test_runs), "four_word_runs_in_training": len(test_runs & train_runs)}


leak_check()
rng.shuffle(train_rows)
manifest = TMP / "train_manifest.jsonl"
manifest.write_text("\n".join(json.dumps({k: r[k] for k in ("audio_filepath", "duration", "text")})
                              for r in train_rows) + "\n")
save_log(train_clips=len(train_rows), train_hours=round(sum(r["duration"] for r in train_rows) / 3600, 2),
         eval_clips={k: len(v) for k, v in evals.items()})

# ---------- scoring ----------
import onnx_asr  # noqa: E402
from nemo.collections.asr.metrics.wer import word_error_rate  # noqa: E402


def score(asr) -> dict:
    res = {}
    for name, items in evals.items():
        hyps = []
        for i in range(0, len(items), 8):
            hyps += asr.recognize([a for a, _ in items[i:i + 8]], sample_rate=SR)
        res[name] = round(100 * word_error_rate([normalise(h) for h in hyps], [normalise(t) for _, t in items]), 2)
    return res


log["eval"]["stock"] = score(onnx_asr.load_model(cfg["stock_onnx_model"], quantization=cfg["quantization"]))
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


def export(folder: Path, net) -> None:
    """Write the four files onnx-asr reads, in fp32 and int8, the way the published model was built.

    NeMo's export leaves `net` unfit to train (every layer frozen, the decoder's start token
    dropped), so never pass the model that is about to be trained."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    folder.mkdir(parents=True)
    net.cpu().eval().export(str(folder / "model.onnx"))  # writes encoder-model.onnx (+ .data) and decoder_joint-model.onnx
    for part in ("encoder-model", "decoder_joint-model"):
        # Unsigned weights, as the published int8 files have; signed cost 1.4 to 3.8 points of WER.
        quantize_dynamic(folder / f"{part}.onnx", folder / f"{part}.int8.onnx", weight_type=QuantType.QUInt8)
    vocab = "".join(f"{tok} {i}\n" for i, tok in enumerate([*net.tokenizer.vocab, "<blk>"]))
    published = Path(hf_hub_download(cfg["onnx_repo"], "vocab.txt", local_dir=TMP / "stock")).read_text(encoding="utf-8")
    if vocab.split() != published.split():
        raise SystemExit("token list differs from the published model's; onnx-asr would decode the wrong words")
    (folder / "vocab.txt").write_text(vocab, encoding="utf-8")
    shutil.copy(hf_hub_download(cfg["onnx_repo"], "config.json", local_dir=TMP / "stock"), folder / "config.json")


if cfg.get("check_export"):
    # The untouched model through our export: if this already differs from stock, the recipe is at fault.
    # A second copy from the download cache: the model being trained cannot be copied
    # (NeMo holds an object proxy deepcopy refuses) and must not be exported.
    untouched = nemo_asr.models.ASRModel.from_pretrained(cfg["base_model"], map_location="cpu")
    export(TMP / "untouched", untouched)
    for q in (None, "int8"):
        log["eval"][f"untouched_{q or 'fp32'}"] = score(onnx_asr.load_model("nemo-conformer-tdt", str(TMP / "untouched"),
                                                                             quantization=q))
    save_log(eval=log["eval"])


def data_cfg(bs: int, path: Path, shuffle: bool) -> DictConfig:
    return DictConfig({"manifest_filepath": str(path), "sample_rate": SR, "batch_size": bs, "shuffle": shuffle,
                       "num_workers": 2, "pin_memory": True, "max_duration": cfg["max_duration_sec"],
                       "min_duration": cfg["min_duration_sec"], "trim_silence": False,
                       "use_start_end_token": False, "use_lhotse": False, "is_tarred": False})


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
    sig, sig_len, tr, tr_len = (t.cuda() for t in next(iter(model._setup_dataloader_from_config(
        data_cfg(bs, probe, False)))))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.0)
    # The fused joint clears its loss reduction while it runs and restores it after; an
    # out-of-memory mid-run skips the restore, and training then gets one loss per clip.
    reduction = model.joint.loss.reduction if model.joint.fuse_loss_wer else None
    try:
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
        ok = True
    except torch.cuda.OutOfMemoryError:
        ok = False
    finally:
        if model.joint.fuse_loss_wer:
            model.joint.loss.reduction = reduction
    peak = torch.cuda.max_memory_allocated() / 2**30
    del opt
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


hours = cfg["max_train_hours"]
trainer = pl.Trainer(devices=1, accelerator="gpu", precision=cfg["precision"], max_steps=cfg["max_steps"],
                     max_time=f"00:{int(hours):02d}:{int(hours % 1 * 60):02d}:00", accumulate_grad_batches=accumulate,
                     gradient_clip_val=1.0, limit_val_batches=0, num_sanity_val_steps=0, logger=False,
                     enable_checkpointing=False, enable_progress_bar=False, log_every_n_steps=10,
                     callbacks=[Progress()])
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
save_log(batch_size=bs, accumulate=accumulate, steps_done=trainer.global_step, training=stopped)

# ---------- export: only the files for the chosen precision go to the output (null = full precision) ----------
export(TMP / "tuned", model)
dest = OUT / cfg["output"]
dest.mkdir()
for f in (TMP / "tuned").iterdir():
    # fp32 keeps every non-int8 file, so the encoder's separate weights file comes along whatever it is named
    if f.name in ("vocab.txt", "config.json") or (".int8." in f.name) == bool(cfg["quantization"]):
        shutil.copy(f, dest / f.name)
save_log(exported=sorted(p.name for p in dest.iterdir()))

log["eval"]["tuned"] = score(onnx_asr.load_model("nemo-conformer-tdt", str(dest), quantization=cfg["quantization"]))
if cfg.get("check_export"):
    # Full precision too: int8 rounds each weight, so a change far too small to learn anything still tips
    # some weights into the next step. Comparing fp32 to fp32 tells training apart from rounding.
    log["eval"]["tuned_fp32"] = score(onnx_asr.load_model("nemo-conformer-tdt", str(TMP / "tuned")))
    # How far training moved anything: weights by the learning rate, the held BatchNorm averages not at all.
    names, after, change = {n for n, _ in model.named_parameters()}, model.state_dict(), {"weights": 0.0, "buffers": 0.0}
    for k, v in untouched.state_dict().items():
        kind = "weights" if k in names else "buffers"
        change[kind] = max(change[kind], float((after[k].cpu().double() - v.double()).abs().max()) if v.numel() else 0.0)
    log["max_change"] = change
save_log(eval=log["eval"], max_change=log.get("max_change"))
