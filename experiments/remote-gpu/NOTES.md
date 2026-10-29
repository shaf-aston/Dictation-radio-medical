# Remote GPU experiment (isolated, high risk)

Purpose: test free remote compute for radio-dictate WITHOUT touching the main app.
Everything lives in this folder. Delete the folder = experiment gone. Nothing in `src/` should import from here until it proves itself.

## Rules (from Shaf)
- Free only. No card, ever. If a service asks for a card, stop.
- CLI, never MCP.
- Keys are in `.env` here (gitignored). Only read them in one small module; never log or commit.
- Medical audio: check each service's data terms BEFORE sending any real patient audio. Test with synthetic/non-patient audio only.

## Accounts connected (all free, set up 2026-09-21 in the studio project)
| Service | Key in .env | What it gives | Tested |
|---|---|---|---|
| Kaggle | KAGGLE_API_TOKEN | free GPU notebooks ~30h/week, batch (minutes to start, not live). CLI: `kaggle kernels push/status/output` | auth OK, no job run |
| Cloudflare | CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN (Workers AI scope) | REST AI, 10k free units/day, hard stop no bill. Has Whisper models in its catalogue (unverified here) | FLUX image call works; Whisper not tried |
| Lightning.ai | LIGHTNING_API_KEY (sk-lit-…) | Model APIs, ~15 free credits then pay-as-you-go. Card need unconfirmed | key not yet tested |

## Things worth testing (each separate, each optional)
1. Cloudflare Whisper speech-to-text: latency and accuracy vs local faster-whisper on non-patient radiology-style audio.
2. Kaggle GPU batch: run a large Whisper model offline over recorded audio to build a "gold" transcript for measuring the local engine's accuracy.
3. Lightning Model APIs: only if a text model there beats local Ollama at cleaning up report text. Watch the credit balance.

## Measure before adopting
Same audio through local vs remote: word error rate, seconds per audio minute, failures. Live dictation needs answers in well under a second, so remote is unlikely to fit live use; batch/accuracy checking is the realistic use.

## How to plug in later (keeps it swappable)
Add one engine file behind the existing ASR engine seam (see `src/dictation/asr/engines/`, `fallback_engine.py`), default off in config. Failure must fall back to the local engine, so the main path never depends on this.

## Housekeeping
- Keys were pasted into a chat: rotate them when the experiment ends.
- Old Cloudflare token "First" (API Tokens Write scope) should be deleted in the Cloudflare dashboard.

## Fine-tune track (Shaf, 2026-09-21)
Goal: train a radiology-tuned speech model on free cloud GPU, then test THAT model easily and separately from the app.
- Train: Kaggle GPU notebook (fine-tune a Whisper size on non-patient/synthetic radiology audio + transcripts). Output = a model folder, downloaded with `kaggle kernels output`.
- Keep versions separate: `models/<name>-v1/`, `v2/` … each with a small `README` (base model, data used, date, measured error rate). Never overwrite a version. The stock local model stays the default.
- Easy test: one script `compare.py` runs the same audio through stock vs each tuned version and prints word error rate + seconds. No app changes needed.
- Only after a version clearly wins on the comparison does it get offered as an optional engine behind the seam.
- Cloudflare is a side option, not the trainer: it can serve stock Whisper / text models, but tuned-Whisper hosting there is unverified.

## Plan and how to run (built 2026-09-21)
Two halves, kept apart. Run commands, file-by-file flow and glossary: [CORE-FLOW.md](CORE-FLOW.md).
- **Training side** (free Kaggle GPU, 2x T4, internet on, no card): synthetic radiology lines -> spoken by Windows voices (one held out) -> trained on Kaggle -> saved as a NEW `models/<name>-vN/`. Knobs live in `train/<engine>.json` and `settings.json`.
- **Inference side** (no app change): the app's `FasterWhisperEngine(model_path=...)` loads any `models/<name>-vN/model`. `compare.py` scores stock vs every version on the same audio.
- **Rules:** synthetic or own-voice non-patient audio only. Main app untouched until a version wins on unseen-voice WER AND term error without slowing down. Then: one optional setting for a model path, default off.
- **Known limits:** training lines are random term slots, so some are medically odd; one TTS voice family can mislead. Real recordings of Shaf reading non-patient scripts are the strongest next data.

## Results log
- v1 `small-en-rad-v1` (2026-09-21): term error 8.45% -> 4.63%, raw WER 6.82% -> 5.59%, unseen voice 9.20% -> 8.81% (10 clips, too few to trust), speed same. Not adopted. Likely partly inflated: training lines were built from the same lexicon the test reports lean on. Next: more varied text, your own voice reading non-patient scripts, a bigger unseen test.
- v2 `small-en-rad-v2` (2026-09-23): wider templates (13->33 patterns/exams), 900->1500 clips, same held-out voice. Result: raw WER and unseen-voice WER identical to v1 (5.59% / 8.81%); term error WORSE (4.63% -> 5.18%) and WER-after-correction WORSE (4.42% -> 4.81%). Not adopted; v1 stays the better candidate. Takeaway: more synthetic template variety diluted the lexicon-dense phrasing the app's correction stage targets, it did not teach anything new. Synthetic-text scaling is refuted as the next lever -- real voice (the `own` set, still not yet run) is the next real test.
- Pipeline lessons (fixed in code): Kaggle CLI on Windows needs an upload-state folder and one archive (not 900 files); status right after a push shows the old run; export needed the base tokenizer.json.
- Parakeet `parakeet-rad-v1` / `v2` (2026-10-23/24): Parakeet TDT 0.6b v3 (the app's engine) fine-tuned on Kaggle with radiology lines spoken by Windows voices, real medical speech (Kaggle), accented speech (EdAcc) and synthetic medical speech (Hani89). v1 ran out of memory at step 169; v2 capped clips at 15 s and finished all 3000 steps. Kaggle's own held-out scoring, word error: medical 22.31% -> 11.88%, accents 20.75% -> 20.35%, Indian-English clinical 41.97% -> 33.88%. Local compare.py (30 clips each):

| Model | Clean WER | Clean unseen voice | Clean term err | Noise WER | Noise term err |
|---|---|---|---|---|---|
| stock fp32 (what the app runs) | 8.06% | 11.49% | 11.72% | 9.68% | 16.35% |
| stock int8 | 9.10% | 12.83% | 13.62% | 14.49% | 23.16% |
| parakeet-rad-v1 (int8) | 8.90% | 12.64% | 13.62% | 12.48% | 21.53% |
| parakeet-rad-v2 (int8) | 8.06% | 11.49% | 10.63% | 11.50% | 17.17% |

  Same precision, v2 wins everywhere (term error 13.62% -> 10.63% clean, 23.16% -> 17.17% noisy). But the app switched to fp32 stock the same day, which is better in noise than v2's int8 export (16.35% vs 17.17%). The rounding to int8 costs more than the training gained. Not adopted. Next (v3): same run, keep the fp32 export that the kernel already builds and throws away. Caveat: the tts voices and lexicon overlap the training lines, so the clean gain is likely inflated; the accent and Eka Care numbers are the honest ones.
- Real voices, `eka` set (2026-10-24): 140 dictations by real Indian-English speakers reading medical phrases (Eka Care, MIT; never in training, the kernel only scored on Eka shards). compare.py, all 140 clips:

| Model | WER | Term err | s/min |
|---|---|---|---|
| stock fp32 (what the app runs) | 22.90% | 11.35% | 6.4 |
| stock int8 | 38.09% | 32.43% | 7.2 |
| parakeet-rad-v2 (int8) | 30.94% | 24.32% | 7.4 |

  On real accented voices int8 rounding is far worse than on the tts set: it nearly triples term error (11.35% -> 32.43%). At the same precision, v2's training cuts term error by a quarter (32.43% -> 24.32%), so the training does help real voices, but the rounding still swamps it. v3 (fp32 export) is the test that decides it. These numbers are not comparable with `replay.py`'s on the same clips: the two average differently.
- Parakeet v3 `parakeet-rad-v3` (2026-10-24): v2's recipe, fp32 export kept. Judged on real voices only (the tts sets overlap the training lines), every clip through both models, paired 95% ranges over clips. Full table, leak check and the words it fixed and broke: `models/parakeet-rad-v3/README.md`.

| Set | WER change (95% range) | Term err stock -> v3 (95% range of change) |
|---|---|---|
| eka clean | -3.20 (-5.03 to -1.49) | 11.35% -> 7.03%, 21 -> 13 of 185 missed (-8.84 to 0.00) |
| eka noise | -1.00 (-2.59 to +0.56) | 18.38% -> 18.92% (-4.00 to +4.84) |
| eka quiet | -1.07 (-2.71 to +0.54) | 22.70% -> 20.54% (-6.12 to +1.75) |
| libri | +0.33 (-0.06 to +0.77) | no medical terms |

  Promising, not proven; not adopted. The only change that beats luck is WER on clean real voices. The medical-word gain there is 8 words and its range reaches zero; noise and quiet show nothing; general English is 0.33 points worse, a tie at best and just over the 0.3 allowed. No leak: 0 Eka phrases or four-word runs in training, 0 shared EdAcc speakers. Caveat: the synthetic training text came from the app's own word list, the same one term error counts. Next: more real medical words decide it, the user's own voice (`own` set) reading reports; wiring v3 in (one setting, default off) is the user's call after that.

- 2026-10-24: v3 wired in as an optional switch, default still standard. Stored at `data/models/fine_tuned/parakeet-rad-v3` (packed copy, identical output). Picked with "Voice model" in web Settings and the desktop settings bar (setting `parakeet_model`). One model in memory at a time; switching preloads it, first words about 1.7s after a switch either way. Same speed as stock.
- Parakeet v4 `parakeet-rad-v4` (2026-10-29): v3 recipe plus LibriSpeech, 30% noisy copies and PARROT radiology TTS (non-commercial licence). Against v3: clean real voices WER -1.96 (range -3.23 to -0.72), radiology reports WER -3.17 and term error -3.56, never worse anywhere. Against stock (Parakeet v2, a different base): eka WER -4.02 after the app's clean-up (range -5.91 to -2.15), but medical words only 14 → 12 of 185 after clean-up. Noise is unchanged, and general English is +0.44 behind stock after clean-up, over the +0.3 bar (range crosses zero). Speed not reliably measured (RAM-starved PC). Installed as an optional voice; default stays standard. Full table: `models/parakeet-rad-v4/README.md`.
