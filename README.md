# Radiology Dictation (Offline, Windows)

An intuitive radiology dictation app with an offline speech-to-text backend (Whisper via faster-whisper) and a clean PySide6 GUI.

## Features
- **Offline transcription** using faster-whisper (Whisper models via CTranslate2)
- **Simple UI**: Start/Stop, templates, copy, save, clear
- **Medical post-processing** to normalize common radiology terms (UK style defaults)
- **Templates** for common studies (CXR, CT A/P, MRI brain)

## Requirements
- Windows 10/11, microphone
- Python 3.10 or 3.11 (recommended)

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
- From an activated venv:
  ```bat
  python src\app.py
  ```
- Or double-click `run.bat` (it will try to use `.venv` if present).

## Usage Tips
- Choose a **model** in the top bar. Smaller = faster; larger = more accurate.
  - `tiny`/`base` are fast on CPU.
  - `small`/`medium` are slower but more accurate.
- Set **Language** (default `en`).
- Click **Start Recording**, speak clearly, then **Stop**.
- The text appears in the editor after processing. Use **Insert Template** to insert a report scaffold.

## Performance
- CPU-only works out of the box. For NVIDIA GPUs, ensure recent drivers; `faster-whisper` will try GPU if available.
- Recording uses 16 kHz mono WAV for best compatibility.

## Troubleshooting
- Microphone errors: Ensure your input device is enabled in Windows Sound Settings. Close other apps using the mic.
- Import errors: Ensure the venv is activated and `pip install -r requirements.txt` succeeded.
- Slow transcription: Use a smaller model (e.g., `tiny` or `base`).

## Customization
- Add or edit templates in `src/templates/`.
- Adjust medical normalization in `src/postprocess.py` (`MEDICAL_REPLACEMENTS`).

## Notes
- Models are downloaded on first use and cached in your user directory.
- No audio leaves your machine; everything runs locally.
