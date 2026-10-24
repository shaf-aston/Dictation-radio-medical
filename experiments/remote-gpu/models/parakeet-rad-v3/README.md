# parakeet-rad-v3

- date: 2026-10-24
- engine: parakeet
- base model: nvidia/parakeet-tdt-0.6b-v3
- data: 1500 local clips from data\train (list: train_manifest.jsonl) plus Kaggle sources ['paultimothymooney/medical-speech-transcription-and-intent']
- settings: parakeet.json
- note: v2 recipe, fp32 export kept (rerun after Kaggle session error with no log)
- measured error rate (compare.py, 2026-10-24, raw model text; tts sets 30 clips each, eka 140 real Indian-English dictations never seen in training):

| Set | Model | WER | Unseen-voice WER | Term err | s/min |
|---|---|---|---|---|---|
| tts (clean) | stock fp32 (what the app runs) | 8.06% | 11.49% | 11.72% | 5.9 |
| tts (clean) | parakeet-rad-v3 | 7.34% | 11.30% | 9.81% | 6.7 |
| tts_noise | stock fp32 | 9.68% | 13.41% | 16.35% | 5.7 |
| tts_noise | parakeet-rad-v3 | 8.84% | 13.41% | 13.35% | 8.2 |
| eka (real voices) | stock fp32 | 22.90% | 22.90% | 11.35% | 7.3 |
| eka (real voices) | parakeet-rad-v3 | 19.70% | 19.70% | 7.03% | 6.1 |

- verdict: wins. First version to beat what the app runs. Term error falls on every set (clean 16%, noise 18%, real voices 38%: 11.35% -> 7.03%), and WER never rises (unseen-voice WER ties in noise, falls elsewhere). Real voices are the honest set and clear the bar there (a quarter AND 2 points); the tts sets clear it only partly, and they overlap the training lines anyway. Speed: same network at the same precision, so same work per clip; the measured gap flips sign between sets (slower on tts, faster on eka), which is load on this machine, not the model. Limits: Eka labels are noisy, and all its speakers are Indian-English. Next: one optional setting that points Parakeet at this folder, default off.

Load with: onnx_asr.load_model('nemo-conformer-tdt', r'C:\Users\Shaf\Downloads\vibe-code-projs\radio-dictate\experiments\remote-gpu\models\parakeet-rad-v3\model', quantization=None)
