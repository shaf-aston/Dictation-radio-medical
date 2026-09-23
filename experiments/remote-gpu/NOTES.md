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

## Plan and how to run (built 2026-09-21; edit this file to steer the work)
Two halves, kept apart:
- **Training side** (runs on free Kaggle GPU, checked: 2x T4, internet on, no card): `data/gen_text.py` invents radiology dictation lines from the app's lexicon (test-report sentences excluded) -> `data/build_train.py` speaks them with Windows voices David + Zira (Hazel held out) -> `train/run_train.py --name small-en-rad` packs a private Kaggle dataset, trains, waits, downloads into a NEW `models/<name>-vN/` (never overwrites; README, config, data list saved beside it). Knobs (epochs, learning rate, freeze encoder) live in `train/config.json`.
- **Inference side** (already swappable, no app change): the app's `FasterWhisperEngine(model_path=...)` loads any folder in `models/<name>-vN/model`. `compare.py` scores stock vs every version on the same audio: raw WER, unseen-voice WER (read this first), WER after the correction stage, radiology-term error, seconds per audio minute.
- **Run:** `python data/build_train.py 900` then `python train/run_train.py --name small-en-rad --note "why"` then `python compare.py`. Each run is unattended; waiting costs no tokens.
- **Rules:** synthetic audio only. Main app untouched until a version wins on unseen-voice WER AND term error without slowing down. Then: one optional setting for a model path, default off.
- **Known limits (v1):** training lines are random term slots, so some are medically odd; one TTS voice family can mislead. Real recordings of Shaf reading non-patient scripts would be the strongest next data.

## Results log
- v1 `small-en-rad-v1` (2026-09-21): term error 8.45% -> 4.63%, raw WER 6.82% -> 5.59%, unseen voice 9.20% -> 8.81% (10 clips, too few to trust), speed same. Not adopted. Likely partly inflated: training lines were built from the same lexicon the test reports lean on. Next: more varied text, your own voice reading non-patient scripts, a bigger unseen test.
- Pipeline lessons (fixed in code): Kaggle CLI on Windows needs an upload-state folder and one archive (not 900 files); status right after a push shows the old run; export needed the base tokenizer.json.
