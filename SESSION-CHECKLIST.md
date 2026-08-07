# Where things stand

`feat/eval-instrument` is merged into `master` — the two are the same commit.
Work since then is on `claude/eval-instrument-merge-check-j6apr7`.
Tests pass, lint clean, theme check clean.

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

## Landed since — marks, run log, /developer

| What you asked for | Where it landed |
|---|---|
| Show which words can be checked, not just answer one | `medical/term_lookup.py::suspect_terms` + `ui/term_marks.py` + web underlay |
| Tell the user the feature exists | hint under the editor, retires after a few uses |
| Run log per dictation | `features/run_log.py` (capped JSONL, both front-ends) |
| `/developer` diagnostics page | `ui/frontends/developer.*` + `GET /developer`, `GET /api/runs` |
| Long-vs-short, answered with numbers | run-length bands on that page |

**The marks are cyan, not red.** `tokens.json` reserves `rec` for recording and
clinical severity; a possible typo underlined in red would read as a finding.
`glow` is the token for the machine's own suggestions.

**A mark never opens onto an empty popup.** A word is marked only if it is not
standard English, not in the membership wordlist, *and* the curated lexicon
holds something it might have been. A dead-end mark teaches the radiologist to
ignore the next one.

**Nothing is inserted into the report.** Desktop marks are `setExtraSelections`;
the web draws a transparent copy behind the textarea. Verified by rendering both
and asserting the document is byte-for-byte unchanged.

The English-word guard moved from `dictation/postprocess/medical_dict_match.py`
to `medical/medical_dict.py`, so the corrector and the marking scan share one
answer to "is this a real word?". The old private name is kept as an alias.

### Open on the run log

- `run_log_store_text` defaults **on**, so report text sits in `data/runs.jsonl`
  (local, gitignored, capped at 200 runs). Set it false to keep every timing and
  drop only the body — your call.
- The long-vs-short table needs real runs of both lengths before it says
  anything. It is currently an empty frame waiting for use.

## Blocked on you

- **Record the `own` gold set** (~15 min): `python -m scripts.eval.build_sets --set own --record`.
  It is the only measurement made on your actual voice, and every accuracy
  question — including whether Parakeet beats Whisper on anatomy — needs it.
- **Shell direction** for the desktop beyond One Surface (Console / Lightbox),
  if you want to go further than the current tidy.
