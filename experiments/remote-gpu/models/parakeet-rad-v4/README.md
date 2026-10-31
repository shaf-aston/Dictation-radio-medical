# parakeet-rad-v4

- date: 2026-10-28
- engine: parakeet
- base model: nvidia/parakeet-tdt-0.6b-v3
- data: 1500 local clips from data\train (list: train_manifest.jsonl) plus Kaggle sources ['paultimothymooney/medical-speech-transcription-and-intent']
- settings: parakeet.json
- note: v4: v3 recipe + LibriSpeech train-clean-100, PARROT radiology TTS, 30% noisy copies (babble/hiss), two-step batch probe

Load with: onnx_asr.load_model('nemo-conformer-tdt', r'C:\Users\Shaf\Downloads\vibe-code-projs\radio-dictate\experiments\remote-gpu\models\parakeet-rad-v4\model', quantization=None)

Licence: trained on PARROT (CC BY-NC-SA 4.0), so this model is for non-commercial use only.

## Training data (from train_log.json)

16,796 clips, 30.6 hours, all 1 to 15 s. 3,000 steps at batch 2 x 8, finished in 198 min on a Kaggle T4.

| Source | Kept of looked at | What it is | Licence |
|---|---|---|---|
| radiology_synth | 595 of 1,500 | our radiology lines, Windows voices | ours |
| kaggle_medical | 6,000 of 6,085 | real people reading symptom phrases | see dataset |
| edacc | 3,000 of 5,760 | real accented English conversation (validation split) | see dataset |
| hani89 | 38 of 1,328 | synthetic medical speech (most clips too long) | see dataset |
| librispeech | 2,000 of 3,271 | everyday read English, train-clean-100 | CC BY 4.0 |
| parrot | 1,287 of 7,587 | radiologists' reports (MSK included), Kokoro TTS, train split; 83% of pieces were over 15 s | CC BY-NC-SA 4.0 |
| noisy copies | 3,876 | 30% of the above again, with babble or hiss at 0 to 20 dB (not pink: the local noise test uses pink) | |

Not used: `modulate/synthetic-medical-dictation`. Each recording is a whole note well over 15 s, the text
spells numbers out and says "period" aloud, so 0 clips pass the training rules.

## Measured on held-out sets (2026-10-29)

Stock = Parakeet **v2** fp32, greedy (onnx-asr): what "standard" runs when sherpa-onnx is not installed. v4 is built on
**v3**, so stock and v4 are different base models. (Kaggle's own check in train_log.json uses v3 as stock: a different
baseline, on different clips. Don't mix the two.) v3, v4: the same clips through each, raw model text.
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

After the app's clean-up chain (`postprocess_transcript`, medium), stock vs v4 on the same transcripts:

| Set | WER stock | WER v4 | v4 minus stock (95% range) | Term missed stock / v4 of 185 |
|---|---|---|---|---|
| eka | 20.82% | 16.80% | -4.02 (-5.91 to -2.15) | 14 / 12 |
| eka_noise | 27.58% | 27.08% | -0.50 (-2.32 to +1.26) | 34 / 30 |
| eka_quiet | 28.43% | 28.11% | -0.32 (-2.18 to +1.48) | 38 / 33 |
| libri | 2.01% | 2.45% | +0.44 (-0.15 to +1.09) | none |

The WER gain on eka survives clean-up. The medical-word gain mostly doesn't: clean-up already fixes 5 of stock's
7 extra misses on eka (19 → 14), leaving 14 vs 12.

Speed: **not reliably measured.** Each model was timed once (beam 1, models run one after another) on a PC with
2.7 of 14.8 GB RAM free. Seconds per audio minute ranged 6.1 to 8.0 for both on eka_noise, eka_quiet and parrot, but
stock on eka was 37.5 and all three models on libri were 22 to 92 (paging). A block-alternating re-test ran out of
memory. The encoder is the same size as stock's (2.44 GB); the decoder+joint is twice as large (72 vs 36 MB; 8,192 vs
1,024 word pieces). A large difference is unlikely, but it is unproven.

Caveat: parrot's voices are the same Kokoro TTS voices that read the training split (different report pieces;
pieces sharing a sentence with training were dropped). So the parrot gain says how well report wording was learned,
not how the model hears a person. The eka and libri numbers are the honest ones.

Medical words v4 got right that stock missed, and the reverse (raw text). These count *distinct words* across clips, not
errors, so fixed minus broke is not a net gain, and a word can be on both lists:
- eka clean: fixed 9 (avascular, cholelithiasis, fissure, infarct, metatarsal, popliteal, spinal, thrombotic, venous); broke 3 (cholecystitis, malleolus, radius)
- eka noise: fixed 14, broke 8. eka quiet: fixed 14, broke 7.
- parrot: fixed 26 (incl. subchondral, osteoarthritis, articular, herniation, multilevel, foramina); broke 7 (incl. osteophyte, plantar, stenosis, subscapularis)

## Leak check

- Local sets against every training line (`train_texts.txt`): eka 0 of 140 phrases, 5 of 2,342 four-word runs
  (stock wording: "chest x ray pa", "x ray pa view", "the right knee joint", "pain in the right", "in the morning and").
  libri 1 of 100 phrases ("what was that"), 16 of 1,538 four-word runs: LibriSpeech train-clean-100 and test-clean
  share no speakers or books, only everyday English.
- Kaggle's own sets (train_log.json `leak`): Eka 5 of 198 "phrases" found, all single words or pleasantries
  ("clear", "discomfort", "pneumonia", "take care", "thanks"); EdAcc 5 of 199 ("oh wow", "yes yes"...).
  parrot_test 0 of 200 phrases; 25 of 225 test pieces were dropped first for sharing a sentence with training.
- Kaggle "validate" overlaps by design (171 of 177 phrases are in the training list): its score is not used.
- Caveat as v3: the synthetic training lines come from the app's own word list, the one term error counts.

## Limits of these numbers

- Ranges resample clips, but eka's 140 clips come from fewer speakers, so the ranges are somewhat too narrow.
- About 12 comparisons were made; one "beyond luck" at 95% could be chance. eka_quiet term (upper end -0.52) is the
  most fragile. eka WER is the most solid.
- eka has judged every version from v1 to v4, so recipe choices have slowly fitted it. An own-voice set is the first fresh test.
- Not measured: v4 against v2 + radiology hotwords (sherpa-onnx), what "standard" becomes when the full `parakeet`
  extra is installed.

## Verdict: best tuned voice so far. Replaces v3 as the optional choice; default stays standard.

- Beats v3 beyond luck on clean real voices (WER -1.96) and on radiology wording (WER -3.17, term -3.56).
- Never worse than v3 on any set.
- Against stock, the one solid gain is about 4 points fewer word errors on real accented clinicians, and it survives
  clean-up. The medical-word gain shrinks to 2 of 185 after clean-up.
- Misses the agreed bar on general English: +0.49 raw, +0.44 after clean-up, both over +0.3 (ranges cross zero).
  Part of this is probably the v3 base, which the app had rejected for v2 as worse at English.
- Noise is unchanged: the 30% noisy copies did not move eka_noise.
- Speed is unproven (see above).
- Installed at `data/models/fine_tuned/parakeet-rad-v4` (packed copy; 8 of 8 parrot clips gave identical text).

## Weights

The app's installed copy in `data/models/fine_tuned/parakeet-rad-v4/` is the only one kept (same model, merged into 5 files; identical output on 20 test clips). compare.py reads it from there.
