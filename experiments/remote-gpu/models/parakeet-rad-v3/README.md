# parakeet-rad-v3

- date: 2026-10-24
- engine: parakeet
- base model: nvidia/parakeet-tdt-0.6b-v3
- data: 1500 local clips from data\train (list: train_manifest.jsonl) plus Kaggle sources ['paultimothymooney/medical-speech-transcription-and-intent']
- settings: parakeet.json
- note: v2 recipe, fp32 export kept (rerun after Kaggle session error with no log)

## Measured on real voices only (2026-10-24)

Stock fp32 (what the app runs) against v3, the same clips through both, raw model text.
"95% range" is a paired bootstrap over clips (10,000 resamples): if it crosses zero, the change could be luck.
The synthetic tts sets are left out: their voices and vocabulary overlap the training lines.

| Set | WER stock | WER v3 | WER change (95% range) | Term err stock | Term err v3 | Term change (95% range) |
|---|---|---|---|---|---|---|
| eka (140 real Indian-English clips) | 22.90% | 19.70% | -3.20 (-5.03 to -1.49) | 11.35% (21 of 185 missed) | 7.03% (13 missed) | -4.32 (-8.84 to 0.00) |
| eka_noise (same voices, added noise) | 28.88% | 27.88% | -1.00 (-2.59 to +0.56) | 18.38% (34 missed) | 18.92% (35 missed) | +0.54 (-4.00 to +4.84) |
| eka_quiet (same voices, turned down) | 31.47% | 30.41% | -1.07 (-2.71 to +0.54) | 22.70% (42 missed) | 20.54% (38 missed) | -2.16 (-6.12 to +1.75) |
| libri (100 clips, general English) | 1.85% | 2.18% | +0.33 (-0.06 to +0.77) | no medical terms | | |

Medical words v3 got right that stock missed, and the reverse (eka, clean):
- fixed 12: popliteal, cholelithiasis, renal, metatarsal, iliac, ankylosing, chondromalacia, lumbar, spondylolisthesis, malleolus, fissure, venous
- broke 4: rectal, ischemia, spinal, inflammatory

In noise it fixed 7 and broke 8; quiet, fixed 9 and broke 5.

## Leak check

- Test sentences: 0 of 723 distinct Eka phrases appear in the training text, and 0 of 878 four-word runs. The same check on two-word runs finds 85 shared pairs, so the method does catch overlap when it exists.
- Speakers: EdAcc trained on its validation split and was held out on test; 0 of 54 test speakers are among the 56 training speakers. Eka was never trained on.
- The Kaggle "validate" score is not used: its phrases overlap the training phrases.
- Not checked line by line: Hani89 (38 synthetic clips, 0.4% of training).
- Caveat: the synthetic training sentences were written from the app's own medical word list, which is the same list term error counts. That is the vocabulary the app needs, but it favours v3 on exactly that score.

## Verdict: promising, not proven. Not adopted.

- Clear gain: overall accuracy on clean real voices (WER down 3.2 points; the range stays below zero).
- Medical words on clean real voices look better (21 missed down to 13), but that is 8 words out of 185, and the range reaches zero. It does not clear the bar (a quarter AND 2 points) with confidence.
- Noisy and quiet audio: no change that beats luck.
- General English: 0.33 points worse, just over the 0.3 allowed; the range includes zero, so it is a tie at best.
- Speed: same network at the same precision; decode time on libri was 86s stock vs 80s v3, which is machine load, not the model.
- Limits: Eka labels are noisy, all its speakers are Indian-English, and it is short phrases, not reports.
- What would decide it: more real medical words. The user's own recorded voice (`own` set) reading real reports is the test; wiring v3 in as an optional setting (default off) is the user's call after that.

Load with: onnx_asr.load_model('nemo-conformer-tdt', r'C:\Users\Shaf\Downloads\vibe-code-projs\radio-dictate\experiments\remote-gpu\models\parakeet-rad-v3\model', quantization=None)
