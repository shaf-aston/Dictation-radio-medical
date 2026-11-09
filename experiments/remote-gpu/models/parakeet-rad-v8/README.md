# parakeet-rad-v8

- date: 2026-11-06
- engine: parakeet
- base model: nvidia/parakeet-tdt-0.6b-v3
- data: 1500 local clips from data\train (list: train_manifest.jsonl) plus Kaggle sources ['paultimothymooney/medical-speech-transcription-and-intent']
- settings: parakeet.json
- note: v8: v4 noise mix (30% noisy, no reverb) + v7 protocol data; PARROT/Hani89 long clips cut at sentence ends (PARROT 4000, Hani89 3000), Libri 3000, 4000 steps; local test sets decoded greedy + hotwords on Kaggle

Load with: onnx_asr.load_model('nemo-conformer-tdt', r'C:\Users\Shaf\Downloads\vibe-code-projs\radio-dictate\data\models\fine_tuned\parakeet-rad-v8', quantization=None)

Installed 2026-11-06 at `data/models/fine_tuned/parakeet-rad-v8` (install_model.py: 20 clips identical) and selected
as the app's voice model (setting `parakeet_model`, was v4). Licence: trained on PARROT (CC BY-NC-SA 4.0), non-commercial only.

## Training data (from train_log.json)

25,101 clips, 52.9 hours, 4,000 steps at batch 2 x 8 on a Kaggle T4. New: clips over 15 s are cut at sentence ends
(stock model's word times; a piece is kept only if it is re-heard starting and ending on its own words).

| Source | Kept | Notes |
|---|---|---|
| radiology_synth | 1,500 | our radiology + MRI-protocol lines (v7 data) |
| kaggle_medical | 6,000 | real people reading symptom phrases |
| edacc | 3,000 | real accented conversation |
| hani89 | 1,809 | 1,245 long clips cut into 3,174 pieces (416 rejected at the edges) |
| librispeech | 3,000 | train-clean-100 |
| parrot | 4,000 (cap) | 2,147 long reports cut into 3,688 pieces (418 rejected at the edges) |
| noisy copies | 30% | babble or hiss, 0 to 20 dB, no reverb (v4's mix; v5-v7's heavier mix cost real-voice accuracy) |

## Measured locally (2026-11-06)

Decoded on Kaggle (greedy, as the app's onnx-asr engine; "+hw" with radiology hotwords, as the app's sherpa engine) and
scored here. Kaggle's greedy text matched this PC's app decode on 20 of 20 eka clips, word for word.
WER after the app's clean-up; range = paired bootstrap 95%, 10,000 resamples over clips.

| Set | stock | v4 | v8 | v8+hw | v8 minus v4 (95% range) |
|---|---|---|---|---|---|
| protocol hand (100 lines no model trained on) | 11.29% | 9.65% | 6.47% | 5.88% | -3.18 (-5.18 to -1.44) |
| protocol gen (trained wording, new voice) | 14.00% | 13.05% | 3.42% | 3.42% | -9.63 (-11.07 to -8.24) |
| parrot (200 radiology report pieces) | 9.14% | 5.31% | 4.47% | 4.33% | -0.84 (-1.24 to -0.46) |
| eka (140 real Indian-English clips) | 20.82% | 16.80% | 16.94% | 17.51% | +0.14 (-0.76 to +1.06) tie |
| eka_noise (same, pink noise 5 dB) | 27.58% | 27.08% | 25.73% | 27.69% | -1.35 (-2.63 to -0.04) |
| libri (general English) | 2.01% | 2.45% | 2.23% | 2.34% | -0.22 (-0.57 to +0.11) tie |

Medical-word error, eka: stock 10.27%, v4 6.49%, v8 5.41%, v8+hw 4.86%. Protocol hand exact after clean-up, v8:
slash 80%, brackets 73%, "1.5 T" 85%, T1/T2 93% (v4: 76 / 62 / 54 / 93%).

Verdict: adopted. Better than v4 on four sets (range misses zero on protocol hand, parrot, eka_noise), tied on eka and
libri, worse nowhere. Same network as v4, so the same speed. Hotwords on top: best on radiology wording (protocol hand
5.88%) and medical words, about 0.6 points worse on general words of real voices; not switched on.

Backed up 2026-11-09 to the private Kaggle dataset `shafeinhashmi/radio-dictate-parakeet-v8` (the app's default model).
A new install restores it into the folder the app loads from (needs the Kaggle CLI logged in to this account):

    kaggle datasets download shafeinhashmi/radio-dictate-parakeet-v8 -p data/models/fine_tuned --unzip
