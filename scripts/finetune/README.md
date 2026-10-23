# Fine-tuning Parakeet for radiology (Kaggle)

The default recogniser, Parakeet TDT 0.6B v2, hears ordinary English well and
radiology words less well. Measured on the `piper` gold set (2026-10-23), 17%
of lexicon terms are wrong before post-processing and 9.7% after it. Most of
what is left is acoustic (steatosis heard as "stenosis", hypodense as
"hyperdense"). A text rule cannot safely fix a mishearing that lands on another
real word; a model that hears the term correctly can.

## The three steps

```bash
# 1. Build the data on this machine (needs Piper voices; see build_sets --set piper)
python -m scripts.finetune.build_dataset --voices <dir of Piper voices> \
    --out data/finetune/radiology-v1 --text my_past_reports.txt   # --text optional

# 2. Zip it, upload as a Kaggle Dataset, open parakeet_kaggle.ipynb in Kaggle,
#    add the dataset, GPU on, Run All. Download parakeet-radiology-onnx.zip.

# 3. Gate it on YOUR voice before using it:
python -m scripts.eval.run_eval --set own --engine parakeet --label stock
python -m scripts.eval.run_eval --set own --engine parakeet --model <unzipped dir> --label tuned
#    Keep it only if medical-term error is lower. Then in dictation_settings.json:
#    "parakeet_model_dir": "<unzipped dir>"
```

## Why it is built this way

| Choice | Reason |
|---|---|
| Never train on `scripts/eval/resources/report_corpus.txt` | The gold sets read that corpus. Training on it would grade the model on its own homework. |
| 20% of lexicon terms and one voice held out | `test_heldout` then tells memorising the training terms apart from learning to hear new ones. |
| Encoder frozen; decoder and joint trained | Synthetic speech teaches words, not acoustics. Freezing the encoder is the recommended way to use synthetic-only data without losing accuracy on real voices (arXiv 2501.12501, 2206.13240). |
| Small constant learning rate (3e-5) | The aim is to adapt, not overwrite. |
| fp16 | The Kaggle T4 has no bf16. |
| Mild augmentation (pace ±10%, level −12 to 0 dB, noise at 25–40 dB SNR) | A quiet reading room is not silent. Heavier noise would teach conditions the app never sees. |
| Your own past reports via `--text` | The fastest way to cover your phrasing and the terms you actually use. They stay local until you upload the zip. |
| Gate on the `own` set | More synthetic data can make real-voice accuracy worse unless it is mixed with real speech. Better on synthetic speech alone proves nothing. |

## What is verified, and what is not

- **Verified here** (`smoke_test.py`, on CPU with NeMo 3.0.0): every notebook
  cell runs, the exported four files are self-contained, onnx-asr output matches
  NeMo output word for word, and the export loads through the app's own engine
  (`ParakeetEngine(<dir>)`).
- **Not verified here:** the training step itself. NeMo's TDT loss is CUDA-only,
  so it can only run on Kaggle. Whether a given run *improves* accuracy is what
  step 3 decides.
- **Estimated, not measured:** time. The default 3000 steps at batch 8 should
  take about 1–2 hours on a T4. Kaggle gives about 30 GPU hours a week.

## Getting more out of it

- **Your own voice is worth more than any amount of synthetic speech.** Record
  the `own` set (`python -m scripts.eval.build_sets --set own --record`) first:
  it is the gate. With more of your own corrected dictations, a second stage
  that also trains the encoder on real speech only becomes worth trying.
- Add more text with `--text`: your past reports, or published report
  templates.
- Pronunciation check: Piper can mispronounce a rare term, and the model then
  learns that wrong sound. Listen to a few clips of the terms you care most
  about before training.
