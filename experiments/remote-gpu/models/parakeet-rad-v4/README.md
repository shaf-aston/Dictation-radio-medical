# parakeet-rad-v4

- date: 2026-10-28
- engine: parakeet
- base model: nvidia/parakeet-tdt-0.6b-v3
- data: 1500 local clips from data\train (list: train_manifest.jsonl) plus Kaggle sources ['paultimothymooney/medical-speech-transcription-and-intent']
- settings: parakeet.json
- note: v4: v3 recipe + LibriSpeech train-clean-100, PARROT radiology TTS, 30% noisy copies (babble/hiss), two-step batch probe

Load with: onnx_asr.load_model('nemo-conformer-tdt', r'C:\Users\Shaf\Downloads\vibe-code-projs\radio-dictate\experiments\remote-gpu\models\parakeet-rad-v4\model', quantization=None)

Licence: trained on PARROT (CC BY-NC-SA 4.0), so this model is for non-commercial use only.

## Measured on held-out sets (2026-10-29)

Stock fp32 (the app's default), v3 and v4, the same clips through each, raw model text.
"95% range" is a paired bootstrap over clips (10,000 resamples): if it crosses zero, the change could be luck.

| Set | WER stock | WER v3 | WER v4 | v4 minus v3, WER (95% range) | Term missed stock / v3 / v4 | v4 minus v3, term (95% range) |
|---|---|---|---|---|---|---|
| eka (140 real Indian-English clips) | 21.35% | 19.15% | 17.19% | -1.96 (-3.23 to -0.72) | 19 / 13 / 12 of 185 | -0.54 (-3.65 to +2.69) |
| eka_noise (same voices, added noise) | 27.72% | 27.54% | 27.54% | 0.00 (-1.35 to +1.33) | 38 / 35 / 31 | -2.16 (-5.88 to +1.53) |
| eka_quiet (same voices, turned down) | 28.97% | 30.07% | 28.90% | -1.17 (-2.58 to +0.22) | 45 / 38 / 35 | -1.62 (-5.81 to +2.38) |
| libri (100 clips, general English) | 1.63% | 2.18% | 2.12% | -0.05 (-0.52 to +0.43) | no medical terms | |
| parrot (201 radiology report clips) | 9.26% | 7.79% | 4.62% | -3.17 (-3.84 to -2.54) | 76 / 70 / 34 of 1012 | -3.56 (-5.19 to -2.15) |

v4 against stock: eka WER -4.16 (-6.05 to -2.31); parrot WER -4.64 (-6.57 to -3.04), term -4.15 (-6.01 to -2.46);
eka_quiet term -5.41 (-10.56 to -0.52); libri WER +0.49 (-0.11 to +1.16).
Speed: 6.3 s per audio minute on eka, the same as stock.

Caveat: parrot's voices are the same Kokoro TTS voices that read the training split (different report pieces;
pieces sharing a sentence with training were dropped). So the parrot gain says how well report wording was learned,
not how the model hears a person. The eka and libri numbers are the honest ones.

## Verdict: best tuned voice so far. Replaces v3 as the optional choice; default stays standard.

- Beats v3 beyond luck on clean real voices (WER -1.96) and on radiology wording (WER -3.17, term -3.56).
- Never worse than v3 on any set.
- Noise is unchanged: the 30% noisy copies did not move eka_noise. General English is about 0.5 points behind stock;
  the range touches zero, so it is still a tie at best.
- Installed at `data/models/fine_tuned/parakeet-rad-v4` (packed copy; 8 of 8 parrot clips gave identical text).
