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

**Two things this harness measures, and they need different runs.**
`run_eval.py` decodes a whole file in one `engine.transcribe()` call: that is
the accuracy instrument, and it never touches `segmenter.py`, `ledger.py` or
`live_session.py`. `replay.py` drives the real `LiveSession` on the schedule
the WebSocket uses and reports **commit latency**, how long a word waits
between being spoken and becoming permanent, alongside WER and term error on
the same clips. Any change to the chunk policy is judged by `replay.py`;
`run_eval.py` cannot see it at all.

The two do not chunk the same way, and it is not a bug in either. Handed a
whole recording, `cut_chunks` takes the *latest* pause up to `soft_max_sec`;
the live loop only ever sees the audio that has arrived, so the same function
takes the *first* pause past `min_sec`. An offline harness therefore measures
a chunk policy the app does not run.

`replay.py` prints `accuracy NOT SCORED` instead of a bare latency figure when
the audio has no reference. A latency number on its own is worthless here: a
policy can always commit sooner by cutting mid-word.

**Every gold set below is continuous speech**, because a speech synthesiser
does not stop to think, so all of them score a pause-related change as zero by
construction. `build_paused_set.py` derives a pause-heavy set from any of them
by inserting real silence at VAD-confirmed mark ends; silence carries no
words, so the source reference stays exactly correct. It measures chunk
*policy*, never acoustics, and is not a substitute for the `own` set.

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

**Two more instruments measure latency with no model and no key**, and
nothing else: `simulate_lag.py` drives the real `LiveSession` with an engine
that sleeps for its declared cost and "hears" synthetic audio whose samples
carry word indices; `web_lag_check.py` does the same through the real web app
and `/ws/dictate`, with `fake_deepgram.py` standing in for Deepgram's live
socket. They can settle a question about the LOOP (chunk plan, streaming,
threads) on any machine. They can never settle an accuracy question.

## Run

```bash
python -m scripts.eval.build_sets --set tts       # synthesise the gold set
python -m scripts.eval.build_sets --set own --record   # record the own set (mic)
python -m scripts.eval.run_eval --set tts --label my-change \
       --baseline data/eval/reports/<earlier>.json

# Chunk-policy work: build a set that contains pauses, then replay it.
python -m scripts.eval.build_paused_set --from tts --gap 6.0
python -m scripts.eval.replay --set tts_paused --label my-change

# A/B a policy without editing dictation_settings.json:
python -m scripts.eval.replay --set tts_paused --trailing-silence 9999 --label baseline
python -m scripts.eval.replay --set tts_paused --chunk-min 4.0 --label shorter-chunks

# --realtime feeds at true speaking pace and reports wall latency instead of
# audio-seconds; use it to see what a loaded machine does to the numbers.

# Loop latency with no model or key (latency only, never accuracy):
python -m scripts.eval.simulate_lag --engine local --final small --seconds 90
python -m scripts.eval.web_lag_check on       # real web app, fake live socket
```
