"""The browser's fallback microphone resampler, run in Node.

Firefox cannot hand a 48 kHz microphone to a 16 kHz audio context, so the page
resamples in its AudioWorklet (src/ui/frontends/resampler.js). Two properties
matter for speech recognition: speech-band tones come through at the right
frequency and level, and energy above the new Nyquist limit is filtered out
rather than folded back into the speech band as aliasing.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

RESAMPLER = Path(__file__).resolve().parents[1] / "src" / "ui" / "frontends" / "resampler.js"

_SCRIPT = """
const { StreamResampler } = require(%(path)s);
function tone(freq, rate, seconds) {
  const n = Math.round(rate * seconds), x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = 0.5 * Math.sin(2 * Math.PI * freq * i / rate);
  return x;
}
function rms(x) { let s = 0; for (const v of x) s += v * v; return Math.sqrt(s / x.length); }
function crossings(x) { let c = 0; for (let i = 1; i < x.length; i++) if ((x[i-1] < 0) !== (x[i] < 0)) c++; return c; }
const out = {};
for (const [name, freq, inRate] of [["speech", 1000, 48000], ["above", 10000, 48000], ["cd", 1000, 44100]]) {
  const r = new StreamResampler(inRate, 16000);
  const x = tone(freq, inRate, 1.0);
  const parts = [];
  for (let i = 0; i < x.length; i += 128) parts.push(r.process(x.subarray(i, i + 128)));
  const y = Float32Array.from(parts.flatMap((p) => Array.from(p))).slice(400, -400);
  out[name] = { samples: y.length + 800, rms: rms(y), crossings_per_sec: crossings(y) / (y.length / 16000) };
}
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def measured():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    script = _SCRIPT % {"path": json.dumps(str(RESAMPLER))}
    done = subprocess.run([node, "-e", script], capture_output=True, text=True, check=True)
    return json.loads(done.stdout)


def test_one_second_in_is_one_second_out(measured):
    for name in ("speech", "cd"):
        assert abs(measured[name]["samples"] - 16000) < 40, name


def test_speech_band_tone_keeps_its_pitch_and_level(measured):
    for name in ("speech", "cd"):
        # 1 kHz crosses zero 2000 times a second; 0.5 amplitude is 0.354 RMS.
        assert abs(measured[name]["crossings_per_sec"] - 2000) < 20, name
        assert abs(measured[name]["rms"] - 0.354) < 0.02, name


def test_energy_above_the_new_nyquist_is_filtered_not_aliased(measured):
    # A 10 kHz tone cannot exist at 16 kHz; without filtering it would fold to
    # 6 kHz at full level. At least 40 dB down is required.
    assert measured["above"]["rms"] < 0.354 * 0.01
