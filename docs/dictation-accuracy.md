# Dictation accuracy — what was measured, and what the numbers said

Recorded 2026-07-28. Companion to [dictation-speed-review.md](dictation-speed-review.md),
which covers latency. This one covers *being right*.

Every number here comes from `python -m scripts.eval.run_eval --set tts --model small.en`
— 30 synthesised radiology reports, 10.8 minutes of audio, the same clips each run.
Synthetic speech, so these are **vocabulary** numbers. They say nothing about what a
real microphone and a real accent do; only the `own` gold set can say that, and it is
still unrecorded.

## The headline

| | WER | medical-term error | false-correction | RTF (median) |
|---|---|---|---|---|
| before (`small.en`, prompt truncated) | 5.5 % | 3.4 % | **11.1 %** | 0.47 |
| after (whole dictionary reaching the decoder) | **4.7 %** | **3.3 %** | **3.9 %** | 0.46 |

**The false-correction rate fell by two thirds** — from 11.1 % to 3.9 %. That is the
number that decides whether the correction layer is worth having at all: it counts the
edits that took a *correct* word and made it wrong. The pipeline now makes 25 true
fixes against 1 false one, a net of +24 on 30 reports.

Nothing about the correction stages changed to achieve that. What changed is that the
decoder finally receives the vocabulary it was always supposed to have (see
[dictation-speed-review.md](dictation-speed-review.md), "81 % of the radiology
dictionary never reaches the decoder"). Whisper transcribes the anatomy correctly in the
first place, so there is less for the fuzzy matcher to guess at — and a guess it never
has to make is a guess it cannot get wrong. **Fixing the input beat filtering the
output.**

### The confound, and why it is not the explanation

The harness started asking the engine for word confidence in the same change, and in
this codebase word timestamps are not inert — the 8-second hallucination gate in
`transcriber.py` keys off them, so they can change the text. That makes "the prompt did
it" a claim worth checking rather than asserting.

Checked directly: the same 12 clips, the same harness, reverting **only** the prompt and
the lexicon to their pre-change versions (`--limit 12`, a partial run used to compare two
configurations against each other, not to score the set).

| 12-clip subset | WER | medical-term error | false-correction |
|---|---|---|---|
| pre-change prompt + lexicon | 3.18 % | 2.88 % | **9.09 %** |
| current | 3.00 % | 4.52 % | **0.00 %** |

The false-correction rate moves with the prompt, not with the harness. The claim holds.

**One honest caveat in the other direction:** medical-term error went *up* on that subset.
Across the full 30 clips it is flat (3.40 % → 3.29 %), so the subset movement is noise
around no change — but the mechanism behind it is real and worth knowing. The old
prompt's surviving tail was generic imaging-physics vocabulary; the new one spends its
223 tokens on musculoskeletal and trauma terms instead. The `tts` set is generic
radiology, so by construction it cannot reward that trade and can only see the cost of
it. Whether the trade is right for the radiologist using this app is a question only the
`own` set can answer, and it is the specialty this app is built around.

## The confidence veto: built, measured, left off

The plan called for gating every correction on the decoder's own per-word confidence, so
a stage could not rewrite a word Whisper was sure about. It is built
(`src/dictation/postprocess/confidence_gate.py`, 24 tests) and it is **shipped
disabled**.

| ceiling | WER | medical-term error | false-correction | true fixes / false | spans blocked |
|---|---|---|---|---|---|
| off | **4.74 %** | **3.29 %** | **3.85 %** | 25 / 1 | — |
| ≥ 0.90 | 4.87 % | 3.84 % | 4.17 % | 23 / 1 | 2 |

The gate fired twice in 30 reports, and **both times it blocked a correction that was
right**. Every metric moved the wrong way. No lower ceiling was worth running: a lower
ceiling protects *more* words, so it can only block more of the same.

The reason is the assumption underneath the design, and the measurement refuted it.
Whisper's confidence does not separate "heard correctly" from "heard wrong" — it is
confidently wrong often enough that a high score is not evidence the word is right. On
synthetic audio, where nearly every word scores high, the signal barely varies at all.

Kept rather than deleted, because the mechanism is sound and costs nothing while off,
and because this instrument cannot give it a fair hearing. The fair test is the `own`
set: a real microphone, a real accent, and confidences that actually vary. One setting
turns it on.

**Not built, on this result:** the worker → UI plumbing that would carry per-word
confidence into the running app. There is no reason to thread a signal through the
ledger, the Qt signals and the post-processing thread for a feature measured to subtract.
`AsrResult` already carries the confidences whenever that changes.

## Acoustic rescoring — rejected on cost, not deferred

Scoring candidate words against the audio for a low-confidence span needs either a
forced-alignment API `faster-whisper` does not expose, or one re-decode per candidate.
On this CPU-only machine `small.en` runs at RTF 0.47 with a fixed ~3.6 s cost per
`transcribe()` call, so N candidate decodes per span breaks the real-time budget
outright. Revisit only behind an engine that exposes frame posteriors — which is exactly
what `AsrEngine.capabilities()` exists to report.

## Two budget collisions worth knowing about

The decoder's prompt slot holds 223 tokens and three things want it:

1. **The shipped vocabulary** — now 210 tokens, sized to fit.
2. **The user's learned terms** — capped at 80 terms, about 216 tokens on real
   radiology words, so a full custom vocabulary would evict the shipped one entirely.
   Fixed by ordering: Whisper keeps the *last* 223 tokens, so the learned terms are
   written first and are the ones dropped when there is no room. Pinned by
   `tests/test_prompt_budget.py`.
3. **Previously decoded text**, when `condition_on_previous_text=True`. The
   confidence-targeted polish pass sets this, so within one call each decoded segment
   pushes the prompt further out of the window. **Open, not fixed** — the live path
   already sets it `False`, so the two paths disagree, but the eval harness decodes
   one-shot and never exercises the polish path, so there is no way to measure which
   setting is better. Changing an accuracy knob that cannot be measured is guessing.

## What would move the needle next

Ranked by expected value, given everything above:

1. **Record the `own` gold set** (~15 min: `python -m scripts.eval.build_sets --set own
   --record`). Every remaining question is blocked on it. It is the only instrument that
   measures real acoustics, and it is the only fair test of the confidence veto.
2. **A better engine (M3).** WER 4.7 % on clean synthetic audio is close to the useful
   floor for this model size; real dictation will be worse. An engine change is the only
   lever left that is worth multiple points.
3. **Settle the `condition_on_previous_text` disagreement**, which needs the harness to
   be able to drive the polish path.
