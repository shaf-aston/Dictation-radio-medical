"""
# Terminal 1: Web app on 127.0.0.1:8005
python -m src.ui.web_app

# Terminal 2: Desktop GUI (separate window)
python -m src.ui
"""

from __future__ import annotations

from src.ui.main_window import main

if __name__ == "__main__":
    main()
