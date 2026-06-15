# Correction system — how spelling fixes scale

Whisper mishears words ("plural effusion" for *pleural effusion*, "coalescistic"
for *cholecystitis*). This document is the **one repeatable path** for fixing a
mishearing so it stays fixed and anyone — developer or radiologist — can add one.

There are three layers, in order of how much work they need from a human —
**most simple misspellings are fixed automatically by the third, with no rule
at all.**

| Layer | Who maintains it | Where | When it fires |
|-------|------------------|-------|---------------|
| Hand-tuned regex table | developers | [`terminology.py`](../src/dictation/postprocess/terminology.py) `_RULES` | complex, order-sensitive rules (lookaheads, whitespace-split acronyms, lambdas) |
| **Data-driven rules** | developers **and radiologists** | [`corrections.yaml`](../src/dictation/postprocess/resources/corrections.yaml) (shipped) + `user_corrections.yaml` (per-site) | simple `misheard → correct` fixes, incl. real-word homophones — needs a rule |
| **Automatic spelling corrector** | nobody (data-driven) | [`medical_dict_match.py`](../src/dictation/postprocess/medical_dict_match.py) | any non-word within a typo's edit distance of a real medical term — **no rule needed** |

The first two run inside the terminology stage (the data layer runs last so user
rules win — see [`rules.py`](../src/dictation/postprocess/rules.py)). The third is
stage 7 and is where simple medical misspellings get fixed in bulk.

### How the automatic corrector decides (stage 7)

A word is corrected only when **all** of these hold, so it fixes typos without
rewriting what the radiologist actually said:

1. It is **not** already a valid English word (offline `pyspellchecker` guard) —
   `there`, `around`, `again` are real words and are never touched.
2. It is **not** in the medical wordlist, an acronym, a protected term, or an
   inflected form of one.
3. The nearest medical term is within a **length-scaled edit distance** — 1 edit
   for normal words, 2 for long ones. This is the key: a single typo in
   `vertabra`/`atelactasis`/`osteophite` is ~88–93% on a character ratio, so the
   old flat-94% floor missed the whole class. Edit distance catches them.

If `pyspellchecker` is not installed, the stage falls back to the old
conservative ratio-only behaviour — a missing dep can never *add*
over-correction, only mute the improvement. Regression net:
[`tests/test_fuzzy_spelling.py`](../tests/test_fuzzy_spelling.py).

You only need a **rule** (layer 1 or 2) when the corrector can't help: a mishear
too far from the right word to be an edit-distance typo (`coalescistic` →
`cholecystitis`), or a real-word homophone (`plural` → `pleural`).

## Rule schema

```yaml
- id: pleural-homophone          # short identifier
  pattern: "plural"              # the misheard word (literal); or raw regex if regex:true
  replacement: "pleural"         # the correction
  context_after: [effusion, space, thickening]   # optional: only fire before one of these
  context_before: [the, a]                        # optional: only fire after one of these
  case_sensitive: false          # optional (default false)
  regex: false                   # optional: treat `pattern` as raw regex
  note: "real-word homophone"    # optional: why
```

**Context guards are the safety mechanism.** `plural → pleural` only fires before
an anatomical word, so grammatical "plural form" is never touched. Always add a
guard when the misheard word is also a real English word.

## Adding a correction — the workflow

### Radiologists (no code)
- **Desktop:** Settings → **Correction Rules…** → add a row → Save.
- **Web:** open **`/correction-rules`** → add a row → Save.

Rules save to `user_corrections.yaml` in the data dir and apply on the next
dictation. App updates never overwrite them.

### Developers (shipped rules + regression net)
1. **Find what to add.** Run the miner against real logs:
   ```bash
   python scripts/mine_corrections.py            # ranked gaps + paste-ready YAML
   python scripts/mine_corrections.py --top 40 --min-count 2
   ```
   It reads three signals and surfaces the highest-frequency gaps, ranked, so you
   fix what actually happens:
   - **Post-dictation edits** (`data/analysis/dictation_edits.jsonl`) — the
     strongest signal: words the radiologist *changed after dictation finished*.
   - **Survivors** (`data/analysis/analysis_*.json`) — unrecognised words that
     slipped through the pipeline.
   - **Hand corrections** (`data/learned_corrections.json`) — single-word edits
     the passive learner already captured.

#### Where the edit signal comes from

[`features/edit_tracking.py`](../src/features/edit_tracking.py) snapshots the
dictation output when recording finishes, and when the report is committed
(export / save / clear / close / next recording) it diffs that snapshot against
the delivered text and logs each change. This is how we answer *"was the
dictation itself wrong?"* — a clinician editing the text is direct evidence it
was. It's wired on **both** front-ends (desktop: `MainWindow.flush_dictation_edits`;
web: the `/api/dictation-edits` endpoint), is **local-only**, gated on
`learning_enabled`, stores only changed spans (not whole reports), and is visible
under **Settings → Learning Statistics**.
2. **Add the rule** to [`corrections.yaml`](../src/dictation/postprocess/resources/corrections.yaml).
3. **Add the regression case** — an `in → out` pair (plus a negative "must not
   change" case for homophones) to
   [`tests/corpus/corrections.yaml`](../tests/corpus/corrections.yaml).
4. **Prove it:**
   ```bash
   python -m pytest tests/test_correction_corpus.py -q
   ```

The corpus is the safety net: every fix ever made is asserted against the full
pipeline, so a later change can never silently regress one.

## Why not a neural spell-checker?

NeuSpell/BERT-style contextual correctors need torch/transformers — heavy deps
that conflict with this app's offline, lazy-import architecture — and aren't
radiology-tuned. So the design splits the problem by error class:

- **Non-word misspellings** (`vertabra`, `atelactasis`) → the edit-distance
  corrector (stage 7), the SymSpell model: confirm the word isn't English, then
  snap it to the nearest medical term within a typo's edit distance. Offline,
  no rule, no inference latency.
- **Real-word homophones** (`plural → pleural`) → edit distance can't help (both
  are valid words), so the context-guarded rule layer handles them.
- **Far mishears** (`coalescistic → cholecystitis`) → too far for edit distance,
  so a `regex` rule in `corrections.yaml`.

A flat similarity ratio (the old approach) can't tell "1 typo in an 8-letter
word" from "a different word" — edit distance can, which is why the corrector
moved to it.
