# Radio Dictate

Offline speech-to-text workstation for radiologists. Dictate a report, get clean, correctly spelled medical text.

Transcription runs locally with Whisper (`faster-whisper`), so audio and text stay on the machine by default.

## Highlights

- **Live transcription**: text streams in while you speak
- **Radiology-tuned**: about 200 domain terms prime the recogniser
- **Correction pipeline**: removes hallucinations, applies voice commands, punctuation, measurements, terminology, accent fixes, fuzzy matching and learned corrections
- **Medical dictionary**: fuzzy matching that protects real terms from over-correction
- **Learns from you**: picks up your edits and applies them next time
- **Critical findings**: negation-aware detection flags urgent results
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

Requires Python 3.12+. Whisper models download to `~/.cache/huggingface/` on first use.

## Stack

Python, faster-whisper, PySide6, FastAPI with WebSockets, rapidfuzz, SymSpell, python-docx.

## Keyboard shortcuts

| Shortcut | Action |
|---|---|
| F5 / F6 | Start / stop recording |
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

## Customise

| What | Where |
|---|---|
| Templates | add a `.txt` to `src/templates/` |
| Macros | `data/macros.json` (hot-reloaded) |
| Medical terms | `src/resources/medical_terms.txt` |
| Correction rules | `src/dictation/postprocess/` |
| Fuzzy cutoff | `cutoff` in `src/dictation/postprocess/medical_dict_match.py` |

## Troubleshooting

| Problem | Fix |
|---|---|
| Microphone errors | check the input device; close other apps using the mic |
| Import errors | activate the venv, run `pip install -e .` |
| Slow transcription | use a smaller Whisper model (`tiny` or `base`) |
| Over-correction | raise the fuzzy cutoff or protect the term |

## Docs

Module map in [CLAUDE.md](CLAUDE.md), style rules in [CODING_STANDARDS.md](CODING_STANDARDS.md), design notes in [docs/](docs/).
