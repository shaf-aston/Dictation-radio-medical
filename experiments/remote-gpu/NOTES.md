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
| Kaggle | KAGGLE_API_TOKEN | free GPU notebooks ~30h/week, batch (minutes to start, not live). CLI: `kaggle kernels push/status/output` | in use: every training run since 2026-09-21 |
| Cloudflare | CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN (Workers AI scope) | REST AI, 10k free units/day, hard stop no bill. Has Whisper models in its catalogue (unverified here) | FLUX image call works; Whisper not tried |
| Lightning.ai | LIGHTNING_API_KEY (sk-lit-…) | Model APIs, ~15 free credits then pay-as-you-go. Card need unconfirmed | key not yet tested |

## Things worth testing (each separate, each optional; not pursued: Kaggle fine-tuning took over)
1. Cloudflare Whisper ASR: latency and accuracy vs local faster-whisper on non-patient radiology-style audio.
2. Kaggle GPU batch: run a large Whisper model offline over recorded audio to build a "gold" transcript for measuring the local engine's accuracy.
3. Lightning Model APIs: only if a text model there beats local Ollama at cleaning up report text. Watch the credit balance.

## Measure before adopting
Same audio through local vs remote: word error rate, seconds per audio minute, failures. Live dictation needs answers in well under a second, so remote is unlikely to fit live use; batch/accuracy checking is the realistic use.

## How a version plugs into the app
Done (2026-10-24): `install_model.py` packs a version into `data/models/fine_tuned/`, and the `parakeet_model` setting ("Voice model") picks it. The app's own engines load it; nothing in `src/` imports this folder.

## Housekeeping
- Keys were pasted into a chat: rotate them when the experiment ends.
- Old Cloudflare token "First" (API Tokens Write scope) should be deleted in the Cloudflare dashboard.

## Fine-tune track (Shaf, 2026-09-21)
Goal: train a radiology-tuned ASR model on free cloud GPU, then test THAT model easily and separately from the app.
- Train: Kaggle GPU notebook (fine-tune a Whisper size on non-patient/synthetic radiology audio + transcripts). Output = a model folder, downloaded with `kaggle kernels output`.
- Keep versions separate: `models/<name>-v1/`, `v2/` … each with a small `README` (base model, data used, date, measured error rate). Never overwrite a version. The stock local model stays the default.
- Easy test: one script `compare.py` runs the same audio through stock vs each tuned version and prints word error rate + seconds. No app changes needed.
- Only after a version clearly wins on the comparison does it get offered as an optional engine behind the seam.
- Cloudflare is a side option, not the trainer: it can serve stock Whisper / text models, but tuned-Whisper hosting there is unverified.

## Plan and how to run (built 2026-09-21)
Two halves, kept apart. Run commands, file-by-file flow and glossary: [CORE-FLOW.md](CORE-FLOW.md).
- **Training side** (free Kaggle GPU, 2x T4, internet on, no card): synthetic radiology lines -> spoken by Windows voices (one held out) -> trained on Kaggle -> saved as a NEW `models/<name>-vN/`. Knobs live in `train/<engine>.json` and `settings.json`.
- **Inference side** (no app change): the app's own engines load a version (`ParakeetEngine` greedy, `ParakeetHotwordEngine` with hotwords via `sherpa_view.py`; `FasterWhisperEngine` for Whisper versions). `compare.py` scores stock vs every version on the same audio.
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
- Parakeet v5 `parakeet-rad-v5` (2026-10-30): v4 recipe, cleaned. Kaggle held-out WER stock -> v5: medical 12.83 -> 6.87, accents 18.54 -> 18.14, eka 32.18 -> 24.37, parrot 8.80 -> 4.61. README: `models/parakeet-rad-v5/`.
- Parakeet v6 `parakeet-rad-v6` (2026-10-31): v5 + 15% spoken MRI-protocol lines ("right slash left", "open bracket", "one point five tesla"). Kaggle eval vs v5: medical 8.12 vs 6.87 (worse), accents 17.81 vs 18.14. Installed in the app as an option.
  - **Corrected after review (2026-10-31):** the first protocol result (WER +fix 9.72 vs v5 13.22, term err 3.24 vs 7.13) came from template wording v6 trained on, and 72% of its term words were just sagittal/axial/coronal. On the rebuilt set's 100 hand-written lines (wording no model trained on), v6 vs v4: WER +fix -0.71 (range -2.02 to +0.57, a tie), raw WER +2.00 (+0.46 to +3.60, worse), term err +1.78 (crosses zero). Its one real gain on unseen wording: brackets exact 77% vs 62%. On the 150 generated lines it wins (WER +fix -2.96, range -4.23 to -1.74): that is trained wording in a new voice, not generalisation.
  - **Why v6 learned little:** the kernel's `usable()` drops any label with number words and clips >15 s; that removed 973 of 1500 synthetic clips incl. every Tesla line. `data/audit_train.py` now replays those rules locally (and exits non-zero if a feature has <30 clips). Fixed in `gen_text.py`: one protocol sentence per clip, <=26 words, labels in digits as Parakeet writes them (`to_target`; the voice still says the words). Result 1500/1500 kept.
  - **Stale-data trap:** Kaggle mounted the previous dataset version and v6's first run came back identical to v5. `run_train.py` now sends the manifest sha256 and the kernel exits on mismatch.
- `protocol` eval set (rebuilt 2026-10-31 after review; manifest sha256 starts 479327B6A048666C, keep it fixed): `data/build_protocol_eval.py 250`, 100 hand-written lines (real-report wording, not the templates) + 150 template lines with seed 4242, none sharing a normalised sentence with any training set, held-out voice only (one TTS voice: ranges are within this set, not over speakers). **Judge on `--by-source` "hand"**; "gen" = trained wording in a new voice. WER cannot see "/" or "( )": `--features` scores exact slash, bracket text, "1.5 T", T1/T2 after correction. Term error here is mostly sequence names (sagittal/axial/coronal), so it says little about medical vocabulary: use eka for that. "WER unseen" is only meaningful on the tts set.
- App: `apply_field_strength` (after the dictionary stage: "3-tisla" only becomes "3-tesla" there) and the same mishearing list inside `apply_measurement_standardisation` (before the dictionary turns "disla" into "disl") -> "3 T". Its mishearing list came from the first protocol set, so "+fix" field-strength numbers flatter every model equally.
- Whisper training (`train/whisper_kernel.py`, `whisper.json`) removed 2026-11-06: unused since small-en-rad-v2, Parakeet beat it on every measure; restore from git history (commit 8ead131) if ever needed. compare.py still scores `stock-whisper` and restored Whisper versions.
- Disk (2026-10-31): v1, v2, small-en v1/v2 weights archived to private Kaggle dataset `shafeinhashmi/rd-model-archive` (restore line in each README). Installed models live only in `data/models/fine_tuned/` (`install_model.py` packs, checks identical output, removes the experiment copy).
- v7 `parakeet-rad-v7` (2026-10-31): v6 with the fixed data. Against v4: protocol hand WER +fix -2.59 (range -4.60 to -0.80), "1.5 T" exact 54% -> 92%; eka WER +fix +0.96 (+0.04 to +1.94), worse on real voices. Not adopted. Table: its README.
- v8 `parakeet-rad-v8` (2026-11-06): v4's noise mix + v7's protocol data + long PARROT/Hani89 clips cut at sentence ends (25,101 clips, 53 h). Against v4: protocol hand WER +fix -3.18 (range -5.18 to -1.44), parrot -0.84, eka_noise -1.35; eka and libri tie; worse nowhere. **Adopted**, installed and selected as the voice model. Kaggle now decodes every local eval set greedy and with hotwords, so scoring a new version locally takes seconds. Table: its README.
- v9 plan (2026-11-07): v8 + real recorded clinical talk, the only free hand-transcribed medical speech found (full search 2026-11-07: Hugging Face, Kaggle, figshare, Zenodo, PhysioNet, papers). PriMock57 (57 GP consultations, CC BY 4.0, GitHub) and Fareez 2022 (272 OSCE interviews, 55 h, CC0, figshare 16550013), each capped at 4000 clips, whole consultations held out (`primock_held`, `fareez_held`). Labels spot-checked against stock Parakeet: differences were formatting ("OK"/"okay", "39"/"thirty nine") or the model mishearing, plus rare transcriber slips; `talk.Slips` drops a line whose odd word is 2 letters from a dictionary word and unheard ("athsma", "ny"), keeps real drug names (Fexofenadine). Rejected: MultiMed English (labels are YouTube auto-captions), clinical-speech-samples (no audio), pauleyc radiology audio (no licence), MeDial-Speech (non-commercial, some machine transcripts), REFLACX (real radiology dictation but PhysioNet credentialed). Question: do real voices with medical words help eka without losing protocol or parrot.
  - Smoke (2026-11-07): every new path ran on Kaggle; the export check tripped on one 24-clip set (parrot_test 8.67 stock vs 10.06 our export). Our export decodes like the published model: locally, same clips, 200 parrot 9.70 vs 9.68 (193 identical), 250 protocol 27.79 vs 27.79 (245 identical); encoder outputs differ by 1e-5. So the check now gates on the average gap over all held-out sets.
  - **Result `parakeet-rad-v9` (2026-11-07): not adopted.** 33,408 clips, 73 h, 4000 steps. Against v8 (WER +fix): protocol +0.46 (range +0.12 to +0.85, worse), hand +0.35 (tie, raw +0.94 worse), eka 0.00 (tie), parrot +0.04 (tie), eka_noise +0.68 (tie), libri +0.38 (+0.05 to +0.78, worse). Real conversation did not help real voices reading medical words, and diluted dictation wording slightly. Table: its README. The run's scoring step ran out of GPU memory (AdamW state left on the GPU; the kernel now clears it), so these numbers are local decodes of the Kaggle-trained model.

- v11 data ready (2026-11-12): the same 1500 radiology/protocol lines, re-read so two-thirds use 24 Kokoro voices (US/UK, men/women) and one-third the Windows voices; before, only David and Zira read them, the narrowest part of the mix. `data/train_v11/` (3.37 h, 24 clips over 15 s dropped). v8 decodes 40 sampled clips at 0.8% WER, so the audio says the words. Run order: v10 first (averaging only, `data/train`), then v11 = v10's recipe with `--data data/train_v11`, so each change is judged on its own by the bar below.

## What counts as better (2026-11-11, replaces "WER on protocol hand")
Judge by the errors a radiologist must catch by rereading, on real voices: the **meaning** column (side, negation, numbers missed after the app's clean-up; `metrics.meaning_slips`) and medical-term error, on eka, eka_noise and parrot. A new version must cut one with a 95% range that misses zero, and be worse on none. Protocol hand is a formatting check only (slash, brackets, "1.5 T"): even stock misses ~1% of its meaning words, so its WER gains are mostly spoken-formatting words. libri guards general English. The set that would settle it is `own` (his voice, kept on this PC, never uploaded): scripts exist in `data/eval/own/`, no audio recorded yet.
Stock vs v8 on this bar (meaning words wrong after clean-up): eka 16.79% -> 5.22% (-11.57, range -16.85 to -6.46), parrot 8.94% -> 4.34% (-4.60, -7.12 to -2.33), eka_noise 13.06% -> 10.07% (tie), protocol 0.64% -> 0.21% (tie), libri 0 both. Medical-term error falls beyond luck on all four medical sets.
Scoring fixes the same day (both sides, every model): spacing between letters and digits is not an error ("40mg" = "40 mg", "sr500" = "sr 500"), and "by" between two numbers = "x" (said vs written). Before them, a third of eka's "meaning" misses were spacing. Every WER here dropped slightly; older README tables were scored before it.

## How to run a version (checklist)
1. `python data/build_train.py 1500` then `python data/audit_train.py` (must pass).
2. Eval sets, only when one changes: `python data/build_protocol_eval.py 250`, `python data/build_parrot_eval.py models/<adopted>/train_texts.txt`, then `python data/pack_eval.py` (uploads them, with the hotword list and bpe.vocab of that moment, for the kernel to decode).
3. `python train/run_train.py --engine parakeet --name parakeet-rad --smoke` (about 30 min; saves nothing, must pass).
4. `python train/run_train.py --engine parakeet --name parakeet-rad --note "..."` (~4 h on Kaggle T4; saves `models/parakeet-rad-vN` and files Kaggle's decodes into decode_store). If the download step fails, the output stays on Kaggle: `python -c "import sys; sys.path[:0]=['.','train']; from run_train import *; cfg=json.load(open('train/parakeet.json')); cfg.pop('smoke'); fetch_output(OUT, cfg); save_version(next_version('parakeet-rad'), 'parakeet', cfg, <clips>, HERE/'data'/'train', '<note>')"`.
5. `python compare.py --set protocol --features --by-source --only <old> <new> <new>+hw --paired <old> <new>`, and the same with `--set eka` (and parrot, eka_noise, libri); a change counts only by the bar in "What counts as better" above. Seconds, from the stored decodes; `--fresh` decodes on this PC instead (~12 min per model on protocol).
6. Winner: `python install_model.py <name> --remove-source`, pick it under Voice model. Weights never go in git (gitignored); only README / train_log / manifests.
- Weight blend `blend.py` (2026-11-11): stock + 0.75 x (v8 - stock), WiSE-FT, built on this PC in a minute, no training. Against v8: protocol hand raw WER +1.53 (+0.70 to +2.50, worse), eka WER +fix -0.60 (-1.33 to +0.08, tie), parrot +0.27 tie, eka_noise term err +2.16 (+0.49 to +4.57, worse), libri +0.05 (no recovery). Not adopted: v8's gains sit in the change it learned, and pulling back toward stock gives nothing back on general English. A smaller alpha only moves further toward stock. The blend folder was deleted (rebuild: `python blend.py parakeet-rad-v8 0.75`).
- v10 ready, not yet run (2026-11-11): v8's data and seed plus checkpoint averaging (kernel `Average`, 5 points over the last 1000 steps; `average_*` in parakeet.json; primock/fareez caps 0 so only the averaging differs from v8). Smoke passed (averaged steps 15 and 20). The full run was refused: the account's 30 h weekly Kaggle GPU quota was used up (shared with other projects). Rerun step 4 of the checklist when the quota resets. Picking the step by a held-out score was left out: scoring during training is what ran out of GPU memory before, and the cosine schedule ends near zero learning rate.
