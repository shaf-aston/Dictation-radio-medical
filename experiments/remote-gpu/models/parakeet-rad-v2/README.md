# parakeet-rad-v2

- date: 2026-10-24
- engine: parakeet
- base model: nvidia/parakeet-tdt-0.6b-v3
- data: 1500 local clips from data\train (list: train_manifest.jsonl) plus Kaggle sources ['paultimothymooney/medical-speech-transcription-and-intent']
- settings: parakeet.json
- note: v1 ran out of memory after 169 of 3000 steps; clips capped at 15s, memory logged every 50 steps
- result: all 3000 steps, 136 min, GPU memory flat at 10.6-11.5 GB. Kaggle eval (int8, word error): medical 22.31 -> 11.88, EdAcc accents 20.75 -> 20.35, Eka Care 41.97 -> 33.88.
- measured error rate (compare.py, 2026-10-24, 30 clips each, raw model text):

| Model | Clean WER | Clean unseen voice | Clean term err | Noise WER | Noise term err |
|---|---|---|---|---|---|
| stock fp32 (what the app runs) | 8.06% | 11.49% | 11.72% | 9.68% | 16.35% |
| stock int8 | 9.10% | 12.83% | 13.62% | 14.49% | 23.16% |
| parakeet-rad-v1 (int8) | 8.90% | 12.64% | 13.62% | 12.48% | 21.53% |
| parakeet-rad-v2 (int8) | 8.06% | 11.49% | 10.63% | 11.50% | 17.17% |
- verdict: not adopted. Beats stock at the same precision (int8) everywhere, but the app now runs stock at fp32, and in noise fp32 stock still wins (16.35% vs 17.17% term error). Only int8 was exported; the fp32 export is v3.

Load with: onnx_asr.load_model('nemo-conformer-tdt', r'C:\Users\Shaf\Downloads\vibe-code-projs\radio-dictate\experiments\remote-gpu\models\parakeet-rad-v2\model', quantization='int8')

## Weights

Archived 2026-10-31 to the private Kaggle dataset `shafeinhashmi/rd-model-archive` (superseded; removed locally to save disk). Restore into this folder:

    kaggle datasets download shafeinhashmi/rd-model-archive -p . --unzip

then keep `parakeet-rad-v2/model/`. Upload checked: sizes match, encoder sha256 matched on download.
