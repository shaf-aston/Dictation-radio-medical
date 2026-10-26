# parakeet-rad-v1

- date: 2026-10-23
- engine: parakeet
- base model: nvidia/parakeet-tdt-0.6b-v3
- data: 1500 local clips from data\train (list: train_manifest.jsonl) plus Kaggle sources ['paultimothymooney/medical-speech-transcription-and-intent']
- settings: parakeet.json
- note: Parakeet TDT 0.6b v3 fine-tuned on medical, accented and Indian-English clinical speech; BatchNorm held, smoke-gated
- result: stopped out of memory at step 169 of 3000. Kaggle eval (int8, word error): medical 22.31 -> 17.89, EdAcc accents 20.75 -> 20.70, Eka Care 41.97 -> 37.13.
- measured error rate (compare.py, 2026-10-24, 30 clips each, raw model text):

| Model | Clean WER | Clean unseen voice | Clean term err | Noise WER | Noise term err |
|---|---|---|---|---|---|
| stock fp32 (what the app runs) | 8.06% | 11.49% | 11.72% | 9.68% | 16.35% |
| stock int8 | 9.10% | 12.83% | 13.62% | 14.49% | 23.16% |
| parakeet-rad-v1 (int8) | 8.90% | 12.64% | 13.62% | 12.48% | 21.53% |
| parakeet-rad-v2 (int8) | 8.06% | 11.49% | 10.63% | 11.50% | 17.17% |
- verdict: superseded by v2.

Load with: onnx_asr.load_model('nemo-conformer-tdt', r'C:\Users\Shaf\Downloads\vibe-code-projs\radio-dictate\experiments\remote-gpu\models\parakeet-rad-v1\model', quantization='int8')
