# UI decisions

A running record of interface decisions: what was chosen, what was rejected, and
why. One entry per decision, newest first. Keep entries short — the reasoning
matters, the retelling doesn't.

Visual mockups live as published artifacts and are linked per entry. The artifact
is the picture; this file is the record.

---

## 2026-09-01 — The developer console, and the four things the rail got wrong (built)

**Chosen: a drawer under the app, not a second page.** `Ctrl`+`Shift`+`D` (or the
overflow menu) opens a console row beneath the whole app. It is a row rather than an
overlay so it can never cover the microphone button, which is the one control you
still need while reading why the last decode was slow.

It shows two clocks in one stream, told apart by colour: the server's diary
(`src/core/event_log.py`, polled from `GET /api/debug/events?after=<seq>`) and the
browser's own marks. Both are needed. "It feels slow" is a statement about the second
one — when text actually landed on screen — and the cause is nearly always in the
first. Reading either alone gets the wrong answer, which is exactly how a session
earlier that day ended up blaming a Stop button nobody had pressed.

Five numbers sit on the bar: time to first words, gap since the last update, how far
the text is behind the microphone, updates so far, seconds of audio sent. Over budget
they turn amber, never red — `--rec` means recording and clinical severity in this
app, and a slow decode is neither. That needed a new token, `warnText`: `warn` is
tuned to be seen as a filled level bar, and at 12px on the light theme's chrome it
measures 3.6:1, under the 4.5:1 floor. Dimming an already-dim token is how contrast
failures get written; the token file is the contrast contract.

**Rejected: a second WebSocket for diagnostics.** The dictation socket carries the
audio and must not share a connection with anything. A poll that only ever asks for
events newer than the last one it printed costs almost nothing, and the server keeps a
bounded ring anyway — so opening the console after a slow dictation still shows that
dictation, which a socket opened on demand could not.

**Four front-end defects the same pass fixed**, all of them visible only by opening
the app and looking at it:

1. **Quick phrases had no styling at all.** `renderMacros` set `class="macro-chip"`
   and no such rule existed, so a phrase fell through to the generic 999px pill: label
   and sentence run together, centred, wrapped over three ragged lines —
   *"Rotator cuff intactRotator cuff tendons are…"*. They are cards now: name on one
   line, the sentence it inserts quietly under it, clipped to two lines.
2. **The wide-screen `min-height: 0` on the report was losing to the base rule.** Same
   specificity, defined later in the file, so the stage kept a 30rem floor it could not
   shrink below and clipped the microphone row whenever anything else took height.
3. **The report had no head.** The word count did not exist and the "words to check"
   line hung under the editor, right-aligned against nothing.
4. **No `<h1>` on the page**, and the wordmark was `display: none` on phones, so a
   screen reader had nothing to announce the app by at any width.

**How it is held.** `axe-core` over both themes at desktop and phone widths reports
zero violations; `tests/test_event_log.py` covers the three ways the diary could lie
without looking broken (a poll re-sending a line, an unbounded buffer, a block that
fails without recording the time it burned). Neither replaces opening it — every one
of the four defects above was invisible in the code and obvious in a screenshot.

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

**How it is held.** By the rule, not by a checker: colour lives in
`src/ui/tokens.json` and is read only by `src/ui/theme.py`, which renders the Qt sheet
and the web page's custom properties. Neither stylesheet contains a hex value. If you
add one, you have reintroduced the drift this section exists to describe. Note that Qt
reports neither a stylesheet it failed to parse nor a property selector it failed to
match — it just paints something plausible — so a desktop colour change is worth
looking at in the running app rather than trusting by reading.

**One defect that check caught immediately.** The microphone level meter had its
palette baked into Python at import time, so it painted dark-theme colours on the
light theme. Its three states are a Qt property now, resolved by the stylesheet like
every other colour. It was invisible from the code alone — it only showed up by
opening the app on the light theme and looking at the meter.

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
