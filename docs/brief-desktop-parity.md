# Brief: bring the desktop front-end up to the web front-end

From Shaf, 2026-07-28. Delete this file when the work lands.

## The judgement

**The web app is currently the better of the two.** It is the reference. Do not
redesign it to meet the desktop halfway — move the desktop to it.

A proposed third layout ("Lightbox & Margin" — a right-hand margin column that
was to be the only place the software speaks) was **rejected**. Reason: the margin
would sit empty most sessions and steal width from the report to do it. Do not
build it. Do not build a right-hand column at all.

## What "catch up" means

The web app's shape, applied to the desktop:

1. **The report is the subject.** The editor is the brightest surface on screen;
   everything else is quiet chrome around it. Today the desktop gives a
   12-control recording bar (`views.py:338-366`) the same visual weight as the
   report, so the eye has nowhere to land.
2. **Controls that don't change mid-session fold away.** Web keeps Patient,
   Quick phrases, Template and Settings in `<details>` panels that remember
   their state. Desktop should reach the same end: the things you touch while
   dictating stay out; the rest folds.
3. **One coloured control.** Colour already comes from `src/ui/tokens.json`
   (rendered by `src/ui/theme.py` — do not add a hex value anywhere else;
   `scripts/verify_theme.py` will fail you). Red means recording or clinical
   severity. Nothing else gets a colour.

## Critical findings — the constraint that matters

**Thin gutter marks only. The radiologist's own words are never touched.**

- A narrow strip beside the editor carries one mark per finding.
- No underline, no highlight fill, no coloured background, no re-styling of any
  character the radiologist typed. Their words are the thing being judged; the
  software does not write on them.
- Clicking a mark may scroll to and select the phrase. That is the whole
  interaction.
- The count belongs somewhere always-visible so nothing is hidden — but the
  *marking* is the only thing this brief authorises in the text column, and the
  answer is: none.

## The real defect to close while you are in there

Acknowledgement on the desktop is an **event**, not a state, and it leaks:

- `recording_session.py:382` fires the check only when transcription stops —
  before the impression is written — and never again.
- `main_window.py:482-484` writes the report to the clipboard with **no gate and
  no audit entry**. Save and Export to Word are ungated too.

The web already solved this: every exit funnels through `postGatedReport`
(`app.js`), the server answers 409 one rule at a time, clipboard included. Give
the desktop the same single funnel, and make the acknowledgement a property of
the report ("findings outstanding") rather than a one-shot dialog, so it is still
true at export time.

Do not weaken the gate to simplify the layout work.

## Constraints

- No new colour outside `tokens.json`. No new dependency. No new web font.
- `python scripts/verify_theme.py`, `ruff check src scripts`, and
  `python -m pytest tests/ -q` all pass before you call it done.
- Keep it modular and chunked — layout in `views.py`, state in the session, the
  gate in one place both front-ends' rules can be read from.
- Out of scope: scan-assistant UI, live confidence marking, model warm-up UI.
  They were proposed; none is authorised here.
