# small-en-rad-v2

- date: 2026-09-23
- base model: openai/whisper-small.en
- data: 1500 synthetic clips (Windows David + Zira voices; Hazel held out), wider template set than v1 (33 vs 13 patterns/exams), see train_manifest.jsonl
- settings: config.json
- note: v2 spike: more data, more template variety, same encoder-frozen 3-epoch setup as v1
- measured 2026-09-23 (30 synthetic test reports, stock vs v1 vs this): WER raw 6.82% -> 5.59% (same as v1); unseen voice 9.20% -> 8.81% (same as v1, still only 10 clips); WER after correction 5.39% -> 4.81% (worse than v1's 4.42%); radiology-term error 8.45% -> 5.18% (worse than v1's 4.63%); speed unchanged. Verdict: not adopted, v1 remains the better candidate -- see NOTES.md results log.

Load with: FasterWhisperEngine(model_path=r'C:\Users\Shaf\Downloads\vibe-code-projs\radio-dictate\experiments\remote-gpu\models\small-en-rad-v2\model')
