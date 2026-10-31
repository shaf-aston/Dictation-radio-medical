# parakeet-rad-v6

- date: 2026-10-31
- engine: parakeet
- base model: nvidia/parakeet-tdt-0.6b-v3
- data: 1500 local clips from data\train (list: train_manifest.jsonl) plus Kaggle sources ['paultimothymooney/medical-speech-transcription-and-intent']
- settings: parakeet.json
- note: v6: v5 recipe + 15% spoken MRI protocol lines (slash, brackets, Tesla, sequences)
- measured error rate: not yet. Run `python compare.py` and paste the table here.

Load with: onnx_asr.load_model('nemo-conformer-tdt', r'C:\Users\Shaf\Downloads\vibe-code-projs\radio-dictate\experiments\remote-gpu\models\parakeet-rad-v6\model', quantization=None)

## Result (2026-10-31)

| Held-out set (WER %) | stock | v5 | v6 |
|---|---|---|---|
| kaggle_medical_validate | 12.83 | **6.87** | 8.12 |
| edacc_test (accents) | 18.54 | 18.14 | **17.81** |
| ekacare_en | 32.18 | **24.37** | 24.77 |
| parrot_test (radiology) | 8.80 | **4.61** | 4.71 |

**Verdict: v6 does not replace v5.** Better on accents (-0.3 pp), worse on medical phrases (+1.25 pp), level elsewhere.

What reached training: 527 of 1,500 radiology clips (v5: 595), including 20 "slash" and 16 "open bracket" lines, but **0 "tesla" lines**: the protocol generator joined up to three sentences per clip, and clips over 15 s (`max_duration_sec`) are dropped, which removed nearly every field-strength sentence. Fix for v7: one protocol sentence per clip.

Spot check, held-out voice (Hazel), through the app's post-processing:

| Said | stock | v5 | v6 |
|---|---|---|---|
| right slash left elbow | right/left | right/left | right/left |
| open bracket 4 mm close bracket | (4mm) | (4 mm) | (4 mm) |
| one point five tesla magnet | 1.5 Disl magnet | 1.5 distiller magnet | 1.5 distillar magnet |
| three tesla scanner | three-tisla | 3 T | 3-tesla |
| coronal PD fat saturated | PDF saturated | PDF saturated | PD fat-saturated (1 of 4) |

Slash and brackets are heard by every model, so the post-processing rules carry them. "Tesla" is still misheard by all: needs v7's short sentences.

First attempt at this run trained on the previous dataset version (Kaggle mounted it before the new upload finished processing) and came back bit-identical to v5. `run_train.py` now sends the manifest's sha256 and the kernel stops on a mismatch.
