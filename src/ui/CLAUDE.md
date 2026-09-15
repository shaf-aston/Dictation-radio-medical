# src/ui: reading the diagnostics while you dictate

Loads only when work touches `src/ui/`. The always-loaded rules for this
project are in the root [CLAUDE.md](../../CLAUDE.md); this file holds the part
that only matters when you are inside the front-end.

`core/perf.py` is the evidence for all of the above: stage timings (count / mean
/ p95 / max) plus point-in-time gauges like `stream.decode_ratio` are logged
when a recording ends and served at `GET /api/debug/perf`. It is in-process
only: nothing is persisted or sent anywhere, so it does not weaken the
offline invariant.

**The developer console** (`Ctrl`+`Shift`+`D` in the browser, or the overflow
menu) is where both of those are read while dictating. It is a drawer under the
app rather than a page you navigate to, because the question it answers, *why
was that slow?*, is asked mid-dictation. It shows two clocks in one stream:

* the **server's diary** (`core/event_log.py`): every decode with its own
  duration, every chunk commit, stop, hand-back and accuracy pass, plus any
  ordinary log line, drained by polling `GET /api/debug/events?after=<seq>` so a
  poll never re-sends a line already printed;
* the **browser's own marks**: microphone granted, socket open, and one per
  *"text actually appeared on screen"*, which is the only latency a radiologist
  feels. These are drawn in a different colour: the two clocks belong to two
  different processes and must never be read as one.

The five numbers on its bar are the ones a slow dictation is judged on: time to
first words, gap since the last update, how far the text is behind the
microphone, update count, and seconds of audio sent. Anything over budget turns
amber (`--warn-text`), never red: `--rec` means recording and clinical
severity in this app, and a slow decode is neither.

Nothing here weakens the offline rule: both endpoints read in-process buffers
and are served on loopback. Nothing is written to disk and nothing is sent
anywhere.
