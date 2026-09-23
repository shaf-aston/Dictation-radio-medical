# small-en-rad-v1

- date: 2026-09-21
- base model: openai/whisper-small.en
- data: 900 synthetic clips (Windows David + Zira voices; Hazel held out), see train_manifest.jsonl
- settings: config.json
- note: first spike: synthetic TTS, encoder frozen, 3 epochs
- measured 2026-09-21 (30 synthetic test reports, stock small.en vs this): WER raw 6.82% -> 5.59%; unseen voice 9.20% -> 8.81% (only 10 clips); WER after correction 5.39% -> 4.42%; radiology-term error 8.45% -> 4.63%; speed same (12.5 vs 12.8 s per audio minute). Verdict: not a clear win yet, see NOTES.md

Load with: FasterWhisperEngine(model_path=r'C:\Users\Shaf\Downloads\vibe-code-projs\radio-dictate\experiments\remote-gpu\models\small-en-rad-v1\model')

- export repair: tokenizer.json replaced by the base model's (identical vocabulary; the Kaggle-written one was unreadable locally). Weights untouched.
