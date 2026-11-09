# parakeet-rad-v9

- date: 2026-11-07
- engine: parakeet
- base model: nvidia/parakeet-tdt-0.6b-v3
- data: 1500 local clips from data\train (list: train_manifest.jsonl) plus Kaggle sources ['paultimothymooney/medical-speech-transcription-and-intent']
- settings: parakeet.json
- note: v8 recipe + real recorded clinical talk (PriMock57 CC BY 4.0, Fareez OSCE CC0), transcriber slips dropped, whole consultations held out. Kaggle scoring after training ran out of GPU memory (fixed in the kernel since); eval sets decoded locally by compare.py.
- measured error rate (compare.py greedy, v9 minus v8, points; 95% range; 2026-11-07):

| set | v8 WER +fix | v9 WER +fix | change | range |
|---|---|---|---|---|
| protocol (250) | 4.21 | 4.67 | +0.46 | +0.12 to +0.85 (worse) |
| protocol hand (100) | 6.47 | 6.82 | +0.35 | -0.36 to +1.24 (tie; raw +0.94 worse) |
| eka, real voices (140) | 16.94 | 16.94 | +0.00 | -0.95 to +1.05 (tie) |
| parrot (200) | 4.47 | 4.50 | +0.04 | -0.26 to +0.32 (tie) |
| eka_noise (140) | 25.73 | 26.41 | +0.68 | -0.55 to +1.93 (tie) |
| libri (100) | 2.23 | 2.61 | +0.38 | +0.05 to +0.78 (worse) |

  Not adopted: better nowhere. The Kaggle scoring step ran out of GPU memory after training, so the model
  was fetched from the run's output and every set decoded on this PC.

Load with: onnx_asr.load_model('nemo-conformer-tdt', r'C:\Users\Shaf\Downloads\vibe-code-projs\radio-dictate\experiments\remote-gpu\models\parakeet-rad-v9\model', quantization=None)
