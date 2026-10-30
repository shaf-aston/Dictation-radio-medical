# parakeet-rad-v5

- date: 2026-10-29
- engine: parakeet
- base model: nvidia/parakeet-tdt-0.6b-v3
- data: 1500 local clips from data\train (list: train_manifest.jsonl) plus Kaggle sources ['paultimothymooney/medical-speech-transcription-and-intent']
- settings: parakeet.json
- note: v5: v4 + real noise/rooms (OpenSLR 28), 50% noisy copies
- measured error rate: not yet. Run `python compare.py` and paste the table here.

Load with: onnx_asr.load_model('nemo-conformer-tdt', r'C:\Users\Shaf\Downloads\vibe-code-projs\radio-dictate\experiments\remote-gpu\models\parakeet-rad-v5\model', quantization=None)
