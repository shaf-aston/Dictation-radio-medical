"""Check that colour lives in exactly one place.

Run: python scripts/verify_theme.py

This guards the one rule that matters here — a hex value hardcoded back into a
stylesheet is how the desktop and web front-ends drifted apart in the first
place, and it is invisible in a diff review.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ui import theme  # noqa: E402

UI = Path(__file__).resolve().parent.parent / "src" / "ui"

# `%23` is a url-encoded '#' inside the inline SVG icons — those are masks whose
# colour comes from `currentColor`, so they are not a palette leak.
HEX = re.compile(r"(?<!%23)#[0-9A-Fa-f]{3,8}\b")

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


# 1. No stylesheet carries its own colour.
for sheet in (UI / "frontends" / "app.css", UI / "styles" / "app.qss"):
    found = HEX.findall(sheet.read_text(encoding="utf-8"))
    check(not found, f"{sheet.name} hardcodes {found} — move it into tokens.json")

# 2. Both themes render, every placeholder filled, and they are really different
#    sheets (a copy-pasted theme would pass every check but this one).
sheets = {name: theme.render_qss(name) for name in theme.THEMES}
for name, qss in sheets.items():
    check("{{" not in qss, f"{name}: app.qss has an unfilled placeholder")
    check(len(HEX.findall(qss)) > 20, f"{name}: rendered sheet has almost no colour in it")
check(len(set(sheets.values())) == len(sheets), "two themes render identically")

# 3. Every token reaches the page, under all four theme selectors.
css = theme.css_variables()
for key in theme.tokens("dark"):
    name = theme.css_name(key)
    check(css.count(f"--{name}:") == 4, f"--{name} is missing from a theme selector")

# 4. The page still has a slot for them, and something still fills it.
check("__THEME_VARS__" in (UI / "frontends" / "app.html").read_text(encoding="utf-8"),
      "app.html lost its __THEME_VARS__ slot")
check("__THEME_VARS__" in (UI / "web_app.py").read_text(encoding="utf-8"),
      "web_app.py no longer fills __THEME_VARS__")

# 5. A placeholder with no matching token must fail loudly, never render empty —
#    an empty colour is an invisible widget, which is far worse than a crash.
stub = UI / "styles" / "_verify_tmp.qss"
stub.write_text("QWidget { color: {{nope}}; }", encoding="utf-8")
real, theme._QSS_FILE = theme._QSS_FILE, stub
try:
    theme.render_qss("dark")
    failures.append("an unknown placeholder rendered silently instead of raising")
except ValueError:
    pass
finally:
    theme._QSS_FILE = real
    stub.unlink()

# 6. The sheets survive contact with real Qt, and the microphone meter shows
#    three distinct states in each theme. Qt reports neither a stylesheet it
#    failed to parse nor a property selector it failed to match — it just paints
#    something plausible — and the test suite stubs PySide6 out entirely, so this
#    is the only thing standing between a broken sheet and the radiologist. The
#    meter was in fact stuck on dark-theme colours until this check was written.
try:
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QProgressBar

    from src.ui.styles import LEVEL_STATES, set_level_state
except ImportError:
    print("note: PySide6 not installed — skipped the painted check (sections 1-5 ran)")
else:
    app = QApplication.instance() or QApplication([])
    for name in theme.THEMES:
        app.setStyleSheet(sheets[name])
        painted = {}
        for state in LEVEL_STATES:
            bar = QProgressBar()
            bar.setObjectName("level_bar")
            bar.setRange(0, 100)
            bar.setValue(80)
            bar.setTextVisible(False)
            bar.setFixedSize(80, 14)
            set_level_state(bar, state)
            bar.show()
            app.processEvents()
            # Inside the filled chunk, clear of the 1px border.
            painted[state] = bar.grab().toImage().pixelColor(20, 7).name()
            bar.close()
        check(len(set(painted.values())) == len(LEVEL_STATES),
              f"{name}: mic meter states are not distinct on screen — {painted}")
        check(set(painted.values()) <= {v.lower() for v in theme.tokens(name).values()},
              f"{name}: mic meter painted a colour that is not a token — {painted}")

if failures:
    print("FAILED:")
    for line in failures:
        print(f"  - {line}")
    raise SystemExit(1)

print(f"OK — colour lives only in tokens.json "
      f"({len(theme.tokens('dark'))} tokens x {len(theme.THEMES)} themes).")
