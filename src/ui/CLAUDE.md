# src/ui: reading the diagnostics while you dictate

Loads only when work touches `src/ui/`. The always-loaded rules for this
project are in the root [CLAUDE.md](../../CLAUDE.md); this file holds the part
that only matters when you are inside the front-end.

`core/perf.py` is the timing evidence for the live-speed rules in the root
CLAUDE.md: stage timings (count / mean / p95 / max) plus point-in-time gauges
like `stream.decode_ratio` (set by both front-ends through
`LiveSession.record_decode_ratio`) are logged when a recording ends and served at `GET /api/debug/perf`. It is in-process
only: nothing is persisted or sent anywhere, so it does not weaken the
offline invariant.

**The developer console** (`Ctrl`+`Shift`+`D` in the browser, or the overflow
menu) is where perf and the event diary are read while dictating. It is a drawer under the
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

## Why the front-end is served no-store

**A dictation owns the report until it is handed back.** In the browser that
is `sessionPhase` (idle → starting → recording → finishing → idle): Record
works only from idle, the editor is read-only, and New, Clear, a template, a
phrase, undo and every exit (Copy/Save/Word) refuse until the hand-back
(`stopped`) arrives. A whole-report write (New, Clear, template) goes through
`replaceReport`: it asks first, stays one undo step away, and gives the page a
new `reportId`. The server keeps findings answers per report id
(`web/state.findings_for`), so an acknowledgement never answers for the next
patient. Both front-ends ask about, write and audit only the findings still
outstanding (`OutstandingFindings.pending`).

**Files the two apps share are changed under a cross-process lock**
(`core/file_lock.py`, `json_store.update_json`): settings, the model
registry and learned corrections each re-read the file and apply only their
own change. Temp audio lives in one folder per running app
(`paths.temp_dir`), and start-up clears only folders whose app has exited.

Front-end files are cached in memory keyed on the file's modification time
(`web/pages._frontend_cache`), and every page and asset is served `no-store`. Both
halves matter: without the mtime key an edit to app.js needed a server restart,
and without `no-store` the browser kept running the previous release's script
against the current server: which is how the page went on using the old
record-then-upload path, and felt many seconds slower, long after live dictation
had landed.
