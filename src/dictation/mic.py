"""When the microphone is worth a warning: one rule for both front-ends.

What cannot be fixed after recording is clipping (the waveform is lost) and a
level so low it sits in the microphone's own hiss. Both front-ends judge the
same thing, the loudest sample, against the same numbers: the desktop's
recorder here, the browser through its bootstrap payload. Measured otherwise
(the desktop used RMS against 0.95 and a moment's silence as "low") the two
warned about different microphones.

No audio library is imported here, so the web server can read it.
"""

from __future__ import annotations

#: A sample this close to full scale has lost its waveform.
CLIP_PEAK = 0.99
#: Still the loudest sample after QUIET_AFTER_SEC (about -30 dBFS): the wrong
#: microphone, or its gain is turned down.
QUIET_PEAK = 0.03
QUIET_AFTER_SEC = 4.0


def as_payload() -> dict:
    """The numbers for the web page."""
    return {"clip_peak": CLIP_PEAK, "quiet_peak": QUIET_PEAK, "quiet_after_sec": QUIET_AFTER_SEC}
