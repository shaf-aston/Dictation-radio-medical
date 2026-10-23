// Streaming resampler for the microphone, used only when the browser cannot
// open a 16 kHz audio context on a microphone stream (Firefox refuses: the
// context must run at the device's own rate). Chrome resamples for us and never
// loads this file.
//
// Windowed-sinc interpolation (96 taps at the input rate: a 48 kHz
// microphone needs that length for the filter to fall off by 8 kHz) with the
// cut-off below the OUTPUT Nyquist
// frequency, so a 48 kHz microphone's energy above 8 kHz is filtered out
// instead of folding back into the speech band as aliasing. Linear
// interpolation, the easy alternative, does not filter at all.
//
// Loaded into the AudioWorklet scope (audioWorklet.addModule) and, for its
// tests, into Node (tests/test_resampler_js.py): hence no DOM and the export
// guard at the bottom.

class StreamResampler {
    constructor(inRate, outRate, halfTaps = 48) {
        this.step = inRate / outRate;            // input samples per output sample
        this.halfTaps = halfTaps;
        // Cut-off at 90% of the output Nyquist, as a fraction of the input rate.
        this.cutoff = 0.9 * Math.min(1, outRate / inRate) * 0.5;
        this.history = new Float32Array(0);      // unread input, with look-behind
        this.pos = halfTaps;                     // next output time, in history samples
    }

    kernel(x) {
        // Low-pass sinc at this.cutoff, Blackman window over +/- halfTaps.
        if (Math.abs(x) >= this.halfTaps) return 0;
        const w = 0.42 + 0.5 * Math.cos(Math.PI * x / this.halfTaps)
            + 0.08 * Math.cos(2 * Math.PI * x / this.halfTaps);
        const arg = 2 * Math.PI * this.cutoff * x;
        const sinc = x === 0 ? 2 * this.cutoff : Math.sin(arg) / (Math.PI * x);
        return sinc * w;
    }

    // Feed a block of input; returns the output samples it completes.
    process(input) {
        const buf = new Float32Array(this.history.length + input.length);
        buf.set(this.history);
        buf.set(input, this.history.length);
        const out = [];
        while (this.pos + this.halfTaps < buf.length) {
            const centre = Math.floor(this.pos);
            let acc = 0;
            for (let k = centre - this.halfTaps + 1; k <= centre + this.halfTaps; k++) {
                acc += buf[k] * this.kernel(this.pos - k);
            }
            out.push(acc);
            this.pos += this.step;
        }
        // Keep what the next outputs still need to look back on.
        const keepFrom = Math.max(0, Math.floor(this.pos) - this.halfTaps);
        this.history = buf.slice(keepFrom);
        this.pos -= keepFrom;
        return Float32Array.from(out);
    }
}

// Each addModule() script is an ES module, so a class declared here is not
// visible to the tap processor loaded after it unless put on the shared scope.
globalThis.StreamResampler = StreamResampler;
if (typeof module !== 'undefined') module.exports = { StreamResampler };
