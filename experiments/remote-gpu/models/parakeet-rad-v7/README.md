# parakeet-rad-v7

- date: 2026-10-31
- engine: parakeet
- base model: nvidia/parakeet-tdt-0.6b-v3
- data: 1500 local clips from data\train (list: train_manifest.jsonl) plus Kaggle sources ['paultimothymooney/medical-speech-transcription-and-intent']
- settings: parakeet.json
- note: v7: v6 data fixed - one protocol sentence per clip, digit labels; 1500/1500 clips pass kernel filters

Load with: onnx_asr.load_model('nemo-conformer-tdt', r'C:\Users\Shaf\Downloads\vibe-code-projs\radio-dictate\experiments\remote-gpu\models\parakeet-rad-v7\model', quantization=None)

## Measured locally (2026-11-06)

Same clips through each model, raw text and after the app's clean-up ("+fix"). Range = paired bootstrap 95%, 10,000 resamples over clips.

| Set | Measure | stock (v2) | v4 | v7 | v7 minus v4 (95% range) |
|---|---|---|---|---|---|
| eka (140 real Indian-English clips) | WER +fix | 20.82% | 16.80% | 17.76% | +0.96 (+0.04 to +1.94) worse |
| eka | medical-word error | 10.27% | 6.49% | 8.11% | +1.62 (-1.23 to +4.85) tie |
| protocol hand (100 lines no model trained on) | WER +fix | 11.29% | 9.65% | 7.06% | -2.59 (-4.60 to -0.80) better |
| protocol hand | WER raw | 29.76% | 24.59% | 24.82% | +0.24 (-2.17 to +2.57) tie |
| protocol hand | exact "1.5 T" / brackets / slash | 54% / 58% / 76% | 54% / 62% / 76% | 92% / 73% / 84% | |
| protocol gen (trained wording, new voice) | WER +fix | 14.00% | 13.05% | 3.58% | -9.47 (-10.92 to -8.09) |

Verdict: not adopted. Wins MRI-protocol dictation, loses on real voices. Likely cause: v5 onward raised noisy copies to 50% and added 30% reverb (v4: 30%, none); Kaggle's own eka score worsened at that same step. v8 goes back to v4's noise mix and keeps v7's protocol data.
