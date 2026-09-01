# scripts/eval: Measuring dictation quality

Follows the root [`CLAUDE.md`](../../CLAUDE.md): this covers only this harness.

Any change to transcription or post-processing is judged by numbers, not by
reading a sample. The harness transcribes a *gold set* (audio + known-correct
text), runs the pipeline, and reports four things:

- **WER**: the general yardstick.
- **medical-term error rate**: WER restricted to `radiology_lexicon.txt` terms.
  A report can post a respectable WER while mangling every anatomical word in
  it; this is the number that catches that.
- **false-correction rate**: of the edits the post-processing pipeline made,
  the share that took a *correct* word and made it wrong. A correction layer
  that fixes 10 words and breaks 12 is worse than none, and this is what says
  which side of that line it is on.
- **real-time factor**: decode seconds per second of audio. Above 1.0 the
  machine cannot keep up with live speech even decoding each second once.

Four gold sets, each measuring something different: never blend them into one
figure. `own` (the user's own voice: the only set that measures real acoustics)
· `tts` (synthesised radiology reports: medical *vocabulary* under
unrealistically clean audio) · `libri` (public-domain read speech: general
English regression + per-engine RTF) · `bench` (the existing `data/bench_audio/`
clips, once hand-corrected).

What the harness has actually decided so far, including the two changes that
moved the false-correction rate and the one that was measured and switched back
off, is in [docs/dictation-accuracy.md](../../docs/dictation-accuracy.md). Read
it before proposing an accuracy change; several obvious ones are already refuted
there.

Sets live in `data/eval/<name>/` (gitignored: the `own` set is the user's
recorded voice). A reference still marked `[UNREVIEWED]` is a machine draft, and
`corpus.load_set` refuses to score against one: grading a model on its own
output produces a flattering number that measures nothing.

`metrics.py` is pure, no I/O, no model, and its normalisation is the single
place scoring rules live. It canonicalises what the pipeline changes *on
purpose* (spoken units → `mm`, hyphen joins) but deliberately leaves spelling
variants (`calibre`/`caliber`) visible, because silently Americanising a British
report is a real change to the radiologist's text.

## Run

```bash
python -m scripts.eval.build_sets --set tts       # synthesise the gold set
python -m scripts.eval.build_sets --set own --record   # record the own set (mic)
python -m scripts.eval.run_eval --set tts --label my-change \
       --baseline data/eval/reports/<earlier>.json
```
