# UI decisions

A running record of interface decisions: what was chosen, what was rejected, and
why. One entry per decision, newest first. Keep entries short — the reasoning
matters, the retelling doesn't.

Visual mockups live as published artifacts and are linked per entry. The artifact
is the picture; this file is the record.

---

## 2026-07-28 — One colour source for both front-ends (built)

**Chosen: one token file, two renderers.** `src/ui/tokens.json` holds 11 named
colours per theme and is the only place a colour is written down. `src/ui/theme.py`
is its only reader: it renders `styles/app.qss` for Qt and `:root` custom properties
for the web page. `styles/dark.qss` and `styles/light.qss` are deleted, and no
stylesheet contains a hex value.

**The problem it solves.** The two front-ends did not share a single colour value.
The desktop was a code-editor theme — record green, stop pink, export blue, three
saturated hues competing for attention, none meaning anything clinically — while the
web app was blue-grey neutrals with red reserved for recording and severity and cyan
for the machine's voice. Same product, two visual identities, and nothing stopping
them drifting further.

**How it is held.** `scripts/verify_theme.py` fails if a hex value reappears in a
stylesheet, if a theme renders identically to the other, if a token is missing from
any of the four theme selectors, or if an unknown placeholder renders silently
instead of raising. It also builds the real Qt widgets off-screen and reads the
painted pixels back, because Qt reports neither a stylesheet it failed to parse nor a
property selector it failed to match — it just paints something plausible.

**One defect that check caught immediately.** The microphone level meter had its
palette baked into Python at import time, so it painted dark-theme colours on the
light theme. Its three states are a Qt property now, resolved by the stylesheet like
every other colour. Nothing in the test suite could have caught this: the suite stubs
PySide6 out entirely, so `verify_theme.py` is currently the desktop UI's only
automated coverage.

**Still open — the structural half.** Closing the colour gap does not make the two
front-ends one product. The desktop still builds a permanent 12-control recording bar
(`views.py:245-366`) while the web app folds everything but the editor away. The
shell directions proposed for that (One Surface / The Console / Lightbox & Margin)
are not built, and the colour argument in that proposal is now out of date.

---

## 2026-07-28 — Five directions for the rest of the app (proposed, not yet built)

**Mockups:** https://claude.ai/code/artifact/3c909d2e-3819-4d13-bffd-10134b215cda

Named directions for the five parts of the app that had no considered design.
Proposed only — nothing here is built yet, so this entry records the thinking, not
a shipped change.

| Area | Direction | The idea in one line |
|---|---|---|
| Critical findings | **The Red Dot** | Marks in the editor's own margin, not a floating dialog — borrowed from the adhesive dot on a film packet |
| Scan assistant | **The Margin Note** | Region outline inside the frame, label outside it and phrased as a question; the abstention is shown as prominently as the finding |
| Recording state | **The Wet Edge** | One column where opacity only ever increases — makes the "committed text never rewrites itself" guarantee visible |
| Model warm-up | **The Tube Warm-up** | Named ritual with a real number, the app usable throughout; the mic is the only disabled control |
| Quick phrases | **The Stamp Block** | A `:trigger` under the caret and a spoken label; region becomes a ranking signal instead of a mode |

**Two things carried into every direction:** cyan is the machine's voice and red is
severity — neither borrows the other's meaning; and each direction states one thing
it must never do, because in this app the failure modes are the design.

**One defect this surfaced, fixed the same day.** Copy is the primary bar action after
the Option A change below, and it wrote to the clipboard with no critical-findings gate
and no audit entry — while Save TXT and Word both had one. A report naming a
pneumothorax could be pasted into the RIS with no warning ever shown. Every path that
lets text leave the web app now runs the same gate (`/api/report/check`), and a test
asserts the whole set of exits so a new route cannot quietly skip it.

**Still open, and it is why the Red Dot is worth building first.** In the desktop app
the check fires from `on_transcription_finished` — when you stop talking, before the
impression is written — and never again at export. The fix is the design itself:
acknowledgement should be a *state of the report* ("dots outstanding"), not an event
bound to one moment.

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
