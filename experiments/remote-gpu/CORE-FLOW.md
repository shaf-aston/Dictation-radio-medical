# Remote GPU experiment: core flow

Fine-tunes Parakeet (the app's default ASR engine) on free Kaggle GPU for radiology wording and accents, then scores it against the stock model on the same audio. Nothing in `src/` imports this folder; delete it and the experiment is gone. Plan, rules, results log and the run checklist: [NOTES.md](NOTES.md).

## Run (from `experiments/remote-gpu/`)

```bash
python data/build_train.py 1500 && python data/audit_train.py          # training audio (Windows voices), checked
python data/pack_eval.py                                                # once, and again when an eval set changes
python train/run_train.py --engine parakeet --name parakeet-rad --smoke     # ~30 min test run, saves nothing
python train/run_train.py --engine parakeet --name parakeet-rad --note "why" # -> models/parakeet-rad-vN/
python compare.py --set protocol --features --by-source --only parakeet-rad-v8 parakeet-rad-vN --paired parakeet-rad-v8 parakeet-rad-vN
python install_model.py parakeet-rad-vN --remove-source                 # winner only: into the app
```

## Flow

1. `data/gen_text.py` invents radiology dictation lines from the app's lexicon (some as spoken MRI protocol wording), skipping any eval sentence; `data/build_train.py` speaks them with every Windows voice except the held-out one and writes `data/train/`.
2. `train/pack.py` turns the clips into one flac tarball; `data/pack_eval.py` uploads the local eval sets once, refusing any real speech without a public licence.
3. `train/kaggle_io.py` is the only file that talks to Kaggle: upload the data, run the kernel with `train/<engine>.json` written into it, wait, download. `run_train.py` ships the kernel's modules (`KERNEL_FILES`) with the data and hashes them with the manifest, so a stale upload stops the run.
4. `train/parakeet_kernel.py` runs ON Kaggle, step by step: data from nine sources (`sources.py`; real recorded clinical talk, PriMock57 and Fareez, from `talk.py`; long clips cut by `long_clips.py`, noisy copies by `augment.py`, text rules from `text_rules.py`), stock score on six held-out sets, training, export (`export_onnx.py`), tuned score, then `eval_decode.py` decodes the local eval sets greedy and with hotwords (`sherpa_view.py` rebuilds the model for sherpa-onnx).
5. `train/run_train.py` files the result into a NEW `models/<name>-vN/` (never overwrites), and Kaggle's decodes into `decode_store.py` as `<v>` and `<v>+hw`.
6. `compare.py` scores each candidate's stored decodes (decoding only what is missing, through the app's own engines) and prints the tables, features and paired 95% ranges.
7. `install_model.py` packs a winning version into the app's `data/models/fine_tuned/`, checking it decodes identically.

## Where things live

| Kind | File |
|---|---|
| Tunable knobs | `train/parakeet.json` (data caps, learning rate, frozen layers, `smoke` overrides), `settings.json` (Kaggle names, eval sets, licences allowed to upload, wait times) |
| Secrets | `.env`, read only by `config.py` |
| Paths, settings loader, repo on sys.path | `config.py` |
| Stored decodes | `decode_store.py` (under the app's `data/cache/`) |

## Words

- **version**: one folder `models/<name>-vN/`; never edited after it is saved.
- **stock**: the untouched model (`stock-parakeet`, `stock-hotwords`, `stock-whisper`), the baseline every version must beat.
- **smoke run**: `--smoke`, a tiny run with near-zero learning rate that proves the pipeline end to end; its tuned score must match stock.
- **eval set**: audio no version trains on (`eka`, `eka_noise`, `protocol`, `parrot`, `libri`); "hand" protocol lines are wording no model has seen.
- **WER**: word error rate; **+fix** = after the app's correction stage, what the user sees.
- **+hw**: the same model decoded with radiology hotwords (sherpa-onnx beam search).
- **term err**: share of radiology lexicon words the model got wrong.
- **unseen**: audio in a voice the model never trained on; printed only for the tts set.
- **s/min**: decode seconds per minute of audio (speed); nan when decoded on Kaggle.
