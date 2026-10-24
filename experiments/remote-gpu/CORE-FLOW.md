# Remote GPU experiment: core flow

Fine-tunes a speech model (Whisper or Parakeet, the app's default engine) on free Kaggle GPU for radiology terms and accents, then scores it against the stock model on the same audio. Nothing here is imported by `src/`; delete the folder and the experiment is gone. Plan, rules and results log: [NOTES.md](NOTES.md).

## Run (from `experiments/remote-gpu/`)

```bash
python data/build_train.py 900                                   # make training audio (Windows voices)
python train/run_train.py --engine parakeet --name parakeet-rad --smoke   # 20-step test run, saves nothing
python train/run_train.py --engine parakeet --name parakeet-rad --note "why"  # -> models/parakeet-rad-vN/
python compare.py                                                # stock vs every version, synthetic test set
python compare.py --set own                                      # same, on your own recorded voice
```

Own-voice set first: `python -m scripts.eval.build_sets --set own --record` (from the repo root).

## Flow

1. `data/gen_text.py` invents radiology dictation lines from the app's lexicon, skipping any line found in the test reports.
2. `data/build_train.py` speaks them with every Windows voice except the held-out one (`settings.json`), writes `data/train/`.
3. `train/pack.py` turns the clips into one flac tarball, so the upload is one request, not hundreds.
4. `train/kaggle_io.py` is the only file that talks to Kaggle: upload the data, run `train/<engine>_kernel.py` on a GPU with `train/<engine>.json` written into it, wait, download.
5. `train/<engine>_kernel.py` runs ON Kaggle. `whisper_kernel.py` exports a CTranslate2 folder. `parakeet_kernel.py` also pulls public medical and accented speech sets, scores stock vs tuned on three held-out sets (in `train_log.json`), and exports the int8 ONNX folder onnx-asr loads.
6. `train/run_train.py` files the result into a NEW `models/<name>-vN/` (never overwrites) with its README, config and clip list.
7. `compare.py` loads each `models/*/model` through the app's own engine (`FasterWhisperEngine`, or `ParakeetEngine` when the folder has `parakeet.json`) and prints one table.

## Where things live

| Kind | File |
|---|---|
| Tunable knobs | `train/whisper.json`, `train/parakeet.json` (data caps, learning rate, frozen layers, `smoke` overrides), `settings.json` (Kaggle names, held-out voice, wait times) |
| Secrets | `.env`, read only by `config.py` |
| Paths, settings loader | `config.py` |

## Words

- **version**: one folder `models/<name>-vN/`; never edited after it is saved.
- **stock**: the untouched model (`stock-whisper` small.en, `stock-parakeet`), the baseline every version must beat.
- **smoke run**: `--smoke`, a tiny cheap run with near-zero learning rate that proves the pipeline end to end; its tuned score should match stock.
- **unseen**: audio in a voice the model never trained on; read this column first.
- **term err**: share of radiology lexicon words the model got wrong.
- **s/min**: decode seconds per minute of audio (speed).
- **held-out voice**: the one Windows voice kept out of training.
