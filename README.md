# Radio Dictate

Offline speech-to-text workstation for radiologists. Dictate a report, get clean, correctly spelled medical text.

Transcription runs locally with Parakeet (sherpa-onnx, biased toward an ~870-term radiology lexicon), with Whisper (`faster-whisper`) behind it, so audio and text stay on the machine by default.

## Highlights

- **Live transcription**: text streams in while you speak
- **Radiology-tuned**: an ~870-term radiology lexicon biases Parakeet as it decodes (Whisper fallback: a ~40-term prompt)
- **Correction pipeline**: removes hallucinations, applies voice commands ("scratch that", "insert <phrase>"), punctuation, measurements, terminology, UK spelling, accent fixes, fuzzy matching, a context real-word fix and learned corrections
- **Medical dictionary**: fuzzy matching that protects real terms from over-correction
- **Learns from you**: picks up your edits and applies them next time
- **Dictate into templates**: text goes in at the cursor; Tab jumps to the next field like `[Right / Left]`, and saying "left" answers it
- **Checks before release**: unfilled fields, wrong side, wrong-sex organs and a missing impression are flagged before the report leaves
- **Critical findings**: negation-aware detection flags urgent results, and records who was told, how and when
- **Templates and macros**: chest, neuro, abdominal, MSK, ultrasound; quick phrases by region
- **Word export**: formatted `.docx` with a patient details table
- **Safe by design**: auto-save backups and an append-only audit log

## Two interfaces

| Interface | Command |
|---|---|
| Desktop (PySide6) | `python -m src.ui` |
| Web (FastAPI) | `python -m src.ui.web_app`, then open http://127.0.0.1:8005 |

Run both from the project root.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Requires Python 3.12+. Models download once into `data/cache/` on first use (safe to delete).

## Stack

Python, sherpa-onnx / onnx-asr (Parakeet), faster-whisper, PySide6, FastAPI with WebSockets, rapidfuzz, SymSpell, python-docx.

## Keyboard shortcuts

| Shortcut | Action |
|---|---|
| F5 / F6 | Start / stop recording (desktop; F5 reloads a web page) |
| F2 | Start or stop recording (desktop and web; a dictation mic's button can send it) |
| Tab / Shift+Tab | Next / previous template field |
| Ctrl+S | Save as text |
| Ctrl+Shift+W | Export to Word |
| Ctrl+T | Load template |
| Ctrl+R | Reload macros |
| Ctrl+D | Dark / light theme |
| Ctrl+P / Ctrl+M | Patient / macros panel |
| Ctrl+] / Ctrl+[ | Font size up / down |

## Voice commands

- Punctuation: "full stop", "comma", "colon", "hyphen", "new line", "new paragraph"
- Fix a word: "correct word X" replaces the previous word with X
- Undo: "scratch that" removes the sentence you are saying, or the one you just finished
- Quick phrase: "insert" and a phrase's name writes the phrase ("insert ACL intact")

Keys are settings (`key_record_toggle`, `key_next_field`, `key_previous_field` in `dictation_settings.json`), so a SpeechMike or PowerMic in keyboard mode can be set to send them.

## Customise

| What | Where |
|---|---|
| Templates | add a `.txt` to `src/templates/` |
| Macros | `data/macros.json` (hot-reloaded) |
| Spelling targets | `src/resources/radiology_lexicon.txt` (what a typo is corrected to); `medical_terms.txt` only marks a word as known |
| Correction rules | `src/dictation/postprocess/` |
| Fuzzy strictness | `_max_edits` in `src/dictation/postprocess/medical_dict_match.py` |

## Troubleshooting

| Problem | Fix |
|---|---|
| Microphone errors | check the input device; close other apps using the mic |
| Import errors | activate the venv, run `pip install -e .` |
| Slow transcription | the developer console shows each decode's time; Whisper sizes only matter without Parakeet |
| Over-correction | add the word to `radiology_lexicon.txt` or a rule to `corrections.yaml`; or use cleanup level Soft |

## Docs

Module map in [CLAUDE.md](CLAUDE.md), style rules in [CODING_STANDARDS.md](CODING_STANDARDS.md), design notes in [docs/](docs/).
