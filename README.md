# MSK Radiology Dictation (Offline, Windows)

Professional dictation workstation for musculoskeletal radiologists. Fully offline speech-to-text powered by Whisper (via faster-whisper) with a PySide6 GUI.

## Quick Start

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src\app.py
```

That's it. The Whisper model downloads automatically on first use.

## Features

- **Offline live transcription** - real-time text streaming while you record, nothing leaves your machine
- **MSK-tuned Whisper prompting** - domain vocabulary primes the model for radiology terminology
- **98K+ medical dictionary** with rapidfuzz fuzzy matching (92% similarity threshold)
- **9-step post-processing pipeline** - spoken punctuation, measurements, MRI sequences, UK spelling, smart capitalisation
- **18 MSK report templates** - knee, shoulder, hip, spine, ankle, wrist, elbow, pelvis, DEXA, X-ray, CT
- **Quick phrases sidebar** - one-click insertion of common findings by body region (11 regions)
- **Word export** - formatted .docx reports with patient info table, section headers, signature line
- **Auto-save** - periodic backup to `autosave/` folder
- **Dark / light themes** - Catppuccin Mocha and iOS-inspired palettes
- **Persistent settings** - window size, model choice, theme, last template all remembered

## Requirements

- Windows 10/11 with a microphone
- Python 3.10 or 3.11 (recommended)
- NVIDIA GPU (optional, for larger models)

## Setup

1. Open a terminal in this folder.
2. Create and activate a virtual environment:
   ```bat
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bat
   pip install -r requirements.txt
   ```

## Run

```bat
python src\app.py
```

Or double-click `run.bat` if present.

## Keyboard Shortcuts

| Shortcut       | Action                        |
|----------------|-------------------------------|
| F5             | Start recording               |
| F6             | Stop recording                |
| Ctrl+S         | Save report as plain text     |
| Ctrl+Shift+W   | Export report to Word (.docx) |
| Ctrl+T         | Load template                 |
| Ctrl+C         | Copy report to clipboard      |
| Ctrl+L         | Clear editor                  |
| Ctrl+N         | New report                    |
| Ctrl+O         | Open report                   |
| Ctrl+D         | Toggle dark / light theme     |
| Ctrl+P         | Toggle patient info panel     |
| Ctrl+M         | Toggle quick phrases panel    |
| Ctrl+]         | Increase font size            |
| Ctrl+[         | Decrease font size            |

## Project Structure

```
src/
  app.py               Main window, menus, recording controls
  transcribe_worker.py  Background thread for live transcription
  transcriber.py        Whisper model loading and transcription API
  postprocess.py        9-step text correction pipeline
  audio.py              Microphone recording (sounddevice + soundfile)
  macros.py             Quick-phrase library (11 body regions)
  medical_dict.py       98K medical term dictionary + fuzzy matching
  report_manager.py     Auto-save, plain-text save, Word export
  settings.py           Persistent JSON settings
  styles.py             Dark and light Qt stylesheets
  templates/            18 MSK report templates (.txt)
  resources/            Medical terms wordlist
```

## Usage Tips

- Choose a **model** in the bottom bar. Smaller = faster; larger = more accurate.
  - `tiny` / `base` - fast on CPU, good for drafts
  - `small` / `medium` - better accuracy, slower
  - `large-v2` / `large-v3` - best accuracy, needs GPU
- Enable **VAD** (voice activity detection) to filter silence (recommended).
- Use **spoken commands** while dictating:
  - "full stop", "comma", "new line", "new paragraph", "colon", "hyphen"
  - "correct word X" - replaces the previous word with X
- Use **quick phrases** on the left sidebar for common findings.
- Load a **template** before dictating to get a structured report scaffold.

## Post-Processing Pipeline

Dictated text passes through nine ordered steps:

1. **Correction commands** - "correct word X" voice edits
2. **Spoken punctuation** - "full stop" to `.`, "comma" to `,`, etc.
3. **Space normalisation** - collapse whitespace, fix punctuation spacing
4. **Measurements** - "5 millimetres" to `5 mm`, "5 by 3 millimetres" to `5 x 3 mm`, degrees to `°`
5. **MRI terminology** - "T 1 weighted" to `T1-weighted`, "hyper intense" to `hyperintense`
6. **MSK corrections** - anatomy, pathology, grading systems, UK spelling (oedema, haemorrhage)
7. **Radiology corrections** - modality names (CT, MRI, X-ray, DEXA)
8. **Dictionary fuzzy matching** - 98K terms with 92% similarity threshold, protected terms prevent over-correction
9. **Smart capitalisation** - sentence starts capitalised, acronyms preserved

## Performance

- Pre-compiled regex patterns at module load
- 10K cached common medical terms for fast fuzzy lookup (< 1 ms per word)
- Live transcription with beam_size=1 for minimal latency
- Auto-detects best compute type (int8 on CPU, float16 on GPU)
- 16 kHz mono WAV recording for optimal Whisper compatibility

## Troubleshooting

- **Microphone errors** - ensure input device is enabled in Windows Sound Settings, close other apps using the mic
- **Import errors** - activate venv and run `pip install -r requirements.txt`
- **Slow transcription** - use a smaller model (`tiny` / `base`) or check logs for compute_type
- **Over-correcting terms** - adjust the fuzzy cutoff in `postprocess.py` (`cutoff=0.92`), or add terms to `_PROTECTED_TERMS`
- **Missing Word export** - run `pip install python-docx`

## Customisation

- Add or edit templates in `src/templates/` (plain .txt files)
- Add quick phrases by editing `src/macros.py`
- Adjust post-processing rules in `src/postprocess.py`
- Tune fuzzy matching confidence: change `cutoff=0.92` (lower = more aggressive)
- Medical dictionary auto-downloads from [glutanimate/wordlist-medicalterms-en](https://github.com/glutanimate/wordlist-medicalterms-en)

## Notes

- Whisper models are downloaded on first use and cached in your user directory.
- No audio or text leaves your machine. Everything runs locally.
- Settings are stored in `dictation_settings.json` at the project root.
- Auto-saved reports go to the `autosave/` folder.
