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

## Not finished

**Parakeet engine bake-off** — task #39. Code is written, lint-clean, 35 tests
pass, but it is **uncommitted** in the working tree:
`src/dictation/asr/engines/parakeet_engine.py`, `tests/test_parakeet_engine.py`,
and edits to `asr/factory.py`, `features/file_manager.py`, `scripts/eval/run_eval.py`.
What remains is the measurement: run both engines over the same gold-set clips
and record WER, medical-term error rate, false-correction rate and real-time
factor per engine in `docs/dictation-accuracy.md`. **Do not switch the default
engine** — the numbers are the deliverable, the choice is the radiologist's.
This is the real answer to "still slow": Whisper `small.en` decodes ~3x slower
than speech on this CPU, and no setting changes that.

**Not started** — tasks #36–38:

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
