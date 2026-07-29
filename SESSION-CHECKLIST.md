# Where things stand

Branch `feat/eval-instrument`, 11 commits ahead of `master`, nothing pushed.
636 tests pass, lint clean, theme check clean.

## Done and committed

| What you asked for | Where it landed |
|---|---|
| Template stopped being overwritten by dictation | cursor anchor + recording-aware insert (`ui/recording_session.py`, `ui/main_window.py`) |
| Words arriving late while talking | preview decode skipped when the machine is behind (`dictation/worker.py`) |
| Long freeze after Stop | final pass moved off the UI thread, per-chunk progress |
| A sign showing what it is doing | state pill beside Record (`ui/views.py`, `ui/styles.py`) |
| Desktop "messy" | One Surface: top bar, editor, three folding panels |
| Highlight a term, see its neighbourhood | `medical/term_lookup.py` + web popover + desktop popup |

Also fixed on the way: the web status dot had colours hardcoded in JavaScript
where the theme check could not see them; the check now reads `app.js` /
`app.html` too. Panels remember their fold state. `ARCHITECTURE.md` matches the
code again.

| Second engine, measured | Parakeet available via `--engine parakeet`, default unchanged |

**The engine verdict** (full table in `docs/dictation-accuracy.md`): Parakeet is
5.2x faster and gets **twice as many anatomical words wrong** (term error 3.44 %
→ 7.12 %), so it is not the default. It also decodes with no radiology
vocabulary at all, which Whisper does get — wiring that in is the experiment
that would settle whether the speed is free.

A speed claim from earlier sessions was wrong and is corrected in that doc:
Whisper `small.en` decodes at **0.57x real time**, not 3x slower than speech.
The live lag came from decoding the same audio repeatedly, which is fixed.

## Not started — tasks #36–38

- Run log: record every dictation run locally (audio length, time to final output,
  decode ratio, model, word count, stage timings, the output text).
- `/developer` page on the web app: a clean table of those runs, output text
  behind a click-to-expand popup so a long report cannot break the layout.
- Long-vs-short verification: the chunk-once design should make cost flat per
  cycle; the run log is what proves it instead of asserting it.

## Blocked on you

- **Record the `own` gold set** (~15 min): `python -m scripts.eval.build_sets --set own --record`.
  It is the only measurement made on your actual voice, and every accuracy
  question — including whether Parakeet beats Whisper on anatomy — needs it.
- **Shell direction** for the desktop beyond One Surface (Console / Lightbox),
  if you want to go further than the current tidy.
