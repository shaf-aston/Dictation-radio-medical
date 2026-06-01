#!/usr/bin/env python
"""Dependency and data-directory check for Radio Dictate.

Run from the project root:  python scripts/verify_setup.py
"""
import sys
from pathlib import Path

OK = "[OK]"
MISSING = "[MISSING]"
WARN = "[WARN]"


def check_dependency(name: str, import_name: str) -> bool:
    try:
        __import__(import_name)
        print(f"{OK} {name}")
        return True
    except ImportError:
        print(f"{MISSING} {name}")
        return False


def main() -> int:
    print("=" * 50)
    print("Radio Dictate - Dependency Check")
    print("=" * 50)

    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    if version >= (3, 10):
        print(f"{OK} Python 3.10+")
    else:
        print(f"{WARN} Python 3.10+ recommended")
    print()

    print("Checking dependencies...")
    deps = [
        ("PySide6", "PySide6"),
        ("faster-whisper", "faster_whisper"),
        ("sounddevice", "sounddevice"),
        ("soundfile", "soundfile"),
        ("numpy", "numpy"),
        ("rapidfuzz", "rapidfuzz"),
        ("python-docx", "docx"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
    ]
    all_ok = all(check_dependency(name, mod) for name, mod in deps)
    print()

    print("Checking data directory structure...")
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    for subdir in ("temp", "autosave"):
        path = data_dir / subdir
        marker = OK if path.exists() else WARN
        suffix = "exists" if path.exists() else "will be created on first run"
        print(f"{marker} data/{subdir}/ {suffix}")

    macros_path = data_dir / "macros.json"
    if macros_path.exists():
        print(f"{OK} data/macros.json")
    else:
        print(f"{MISSING} data/macros.json")
        all_ok = False

    wordlist = project_root / "src" / "resources" / "medical_terms.txt"
    if wordlist.exists() and wordlist.stat().st_size > 0:
        print(f"{OK} src/resources/medical_terms.txt")
    else:
        print(f"{WARN} src/resources/medical_terms.txt missing — will be downloaded on first use")

    print()
    print("=" * 50)
    if all_ok:
        print(f"{OK} All checks passed.")
        print("Start the desktop app:  python -m src.ui")
        print("Start the web app:      python -m src.ui.web_app")
    else:
        print(f"{MISSING} Issues found. Run:  pip install -r requirements.txt")
    print("=" * 50)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
