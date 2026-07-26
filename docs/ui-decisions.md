# UI decisions

A running record of interface decisions: what was chosen, what was rejected, and
why. One entry per decision, newest first. Keep entries short — the reasoning
matters, the retelling doesn't.

Visual mockups live as published artifacts and are linked per entry. The artifact
is the picture; this file is the record.

---

## 2026-07-26 — Dictation screen: dictate first, everything else folds away

**Chosen: Option A.** The editor and microphone take the top of the screen at
full height. Patient details, quick phrases, template and settings become four
labelled panels underneath that open on click and stay how you left them. The
top bar keeps one primary action (Copy); New, Clear, Save TXT, Word and theme
move behind an overflow menu.

**Mockups:** https://claude.ai/code/artifact/a90bbd34-26a8-4765-b3e4-adc4dedf79de

**The problem it solves.** The screen was ordered by how the app was built, not
how it is used. Top to bottom it ran: instruction banner, six patient fields,
transcription settings, quick phrases, template picker, *then* the editor and
the microphone. The one thing a radiologist opens the app to do was seventh.
Nothing could be hidden, so a radiologist who never touches templates still paid
for them on every report.

**Rejected — Option B, two columns.** Dictation left at full height, patient
fields and phrases in a collapsible right rail. Real merit: patient fields stay
visible so nothing is missed at export, and it maps almost free onto the desktop
app, which already builds a horizontal `QSplitter` in `views.py`. Rejected
because it needs width — on a laptop the rail has to drop below the editor, which
lands you at Option A anyway, so Option A is the same answer with less machinery.

**Rejected — Option C, one thing at a time.** A Dictate / Details / Settings
switcher showing one view at a time. Cleanest possible dictation view. Rejected
because quick phrases are tapped *mid-report*, and putting them behind a view
switch makes the most frequent interaction the most expensive one. It was also
the largest rebuild of the three.

**Applies to both front-ends.** Web (`src/ui/web_app.py`) and desktop
(`src/ui/views.py`), so the two stop feeling like different products.

**Carried regardless of the option chosen:**

- **The front-end comes out of Python.** `web_app.py` was 2,312 lines, of which
  1,750 were a single `HTML_TEMPLATE` string holding all the CSS, HTML and JS.
  That is the real reason the interface was hard to change: no highlighting, no
  linting, no components. Split into real files under `src/ui/frontends/`.
- **Panels remember their state**, saved with the other preferences. This is what
  "configurable to needs" means here — you shape the screen by using it, not by
  finding a settings page.
- **One clear action.** Six equal-weight buttons became one primary plus an
  overflow menu.
- **The permanent "How to use" banner goes.** It teaches once, then costs a row
  on every report forever.
- **No control is removed.** Templates, macros, accent and cleanup settings, VAD,
  the critical-findings warning and both export formats all keep working exactly
  as they do now.
