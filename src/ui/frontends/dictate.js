// Live dictation for the web page: microphone → /ws/dictate → the editor.
// Loaded after app.js and shares its globals (editor, sessionPhase, dev, …);
// app.js reads back only previewText, LIVE_SAMPLE_RATE, mic and tlStep, at
// call time. Kept apart because it is the one part with its own lifecycle
// (audio graph, socket, worklet) and the part most often worked on alone.

// ---------------------------------------------------------------------------
// Live dictation.
//
// The microphone streams raw 16-bit PCM at 16 kHz straight to /ws/dictate, and
// text comes back while you are still speaking. It is raw PCM rather than the
// browser's own MediaRecorder output because a webm/opus blob cannot be decoded
// a piece at a time: waiting for the container to close is exactly the pause
// this replaces.
//
// Two kinds of text arrive. `committed` is decoded once, corrected, and final.
// `preview` is a guess about the words still being spoken; it is shown dimmed
// and never enters the undo history, because presenting a guess as settled text
// is the one thing a report editor must not do.
// ---------------------------------------------------------------------------

// From the server, so the capture rate and the decoder's rate are one fact.
const LIVE_SAMPLE_RATE = BOOTSTRAP.live_sample_rate;

let liveSocket = null;
let audioContext = null;
let micStream = null;
let micNode = null;
let committedText = '';       // what the server has frozen this session
let previewText = '';         // the provisional tail, dimmed on screen
// Dictation goes in at the caret. The editor is rebuilt every update as
// the text before the caret + the dictation + the text after it, so the rest of
// a template below the caret is never rewritten. A selected field is replaced.
let baseBefore = '';
let baseAfter = '';
let baseOriginal = '';        // the editor exactly as it was, for Discard
let gapBefore = '';
let gapAfter = '';
let recordStartedAt = 0;
let timerHandle = null;
let handedOverText = null;  // what Stop handed back, to detect edits since
// When Stop was pressed, so the developer console can price the two things
// that follow it separately: the hand-back (should be instant) and the
// accuracy pass behind it (allowed to take as long as it needs).
let stopPressedAt = 0;

// How long Stop waits for the worklet's last partial frame before giving up
// on it: a fraction of a second of speech is not worth a stuck button.
const MIC_FLUSH_TIMEOUT_MS = 50;

// The microphone is listened to from the moment the audio graph exists, but the
// socket takes a moment longer to open. Sound spoken in that gap is held here
// (null whenever no connect is under way), in order, and sent when it opens.
// Capped so a socket that never opens cannot grow memory without limit: past
// the cap the OLDEST sound is dropped, so what is sent stays one unbroken run
// ending at the present instead of a gap in the middle.
const PRE_SOCKET_MAX_SEC = 10;
let pendingPcm = null;
let pendingSamples = 0;

// The worklet only forwards frames. Every decision stays on the main thread, so
// UI work can never block the audio thread. It buffers render quanta (fixed at
// 128 samples by the Web Audio spec) into fixed-size frames before posting, so
// the socket wakes up a few dozen times a second instead of over a hundred.
// 20ms is the largest frame that still sits well under the 0.5s cycle the
// server decodes on, so it adds no lag the radiologist can see, while cutting
// posts from one per render quantum (128 samples) to 50 a second at 16 kHz.
const FRAME_SECONDS = 0.020;
// Derived from the rate actually in force, never written down as a sample
// count: the same LIVE_SAMPLE_RATE is what the AudioContext is asked for
// below, so a change to the server's setting moves both together and the
// frame stays 20ms instead of silently becoming 6.7ms at 48 kHz.
const FRAME_SAMPLES = Math.max(128, Math.round(LIVE_SAMPLE_RATE * FRAME_SECONDS));
const PCM_WORKLET = [
    `const FRAME_SAMPLES = ${FRAME_SAMPLES};`,
    'class PcmTap extends AudioWorkletProcessor {',
    '    constructor(options) {',
    '        super();',
    '        const o = (options && options.processorOptions) || {};',
    '        // Only when the context could not run at 16 kHz (resampler.js).',
    '        this._resampler = o.inRate && o.inRate !== o.outRate',
    '            ? new StreamResampler(o.inRate, o.outRate) : null;',
    '        this._buf = new Float32Array(FRAME_SAMPLES);',
    '        this._len = 0;',
    '        this.port.onmessage = (event) => {',
    '            if (event.data && event.data.command === "flush") {',
    '                this.port.postMessage({ flushed: this._buf.slice(0, this._len) });',
    '                this._len = 0;',
    '            }',
    '        };',
    '    }',
    '    process(inputs) {',
    '        const input = inputs[0] || [];',
    '        let ch = input[0];',
    '        if (ch && input.length > 1) {',
    '            // Mono was asked for; some browsers return stereo anyway.',
    '            ch = ch.map((v, i) => input.reduce((s, c) => s + c[i], 0) / input.length);',
    '        }',
    '        if (ch && this._resampler) ch = this._resampler.process(ch);',
    '        if (ch) {',
    '            let offset = 0;',
    '            while (offset < ch.length) {',
    '                const space = FRAME_SAMPLES - this._len;',
    '                const take = Math.min(space, ch.length - offset);',
    '                this._buf.set(ch.subarray(offset, offset + take), this._len);',
    '                this._len += take;',
    '                offset += take;',
    '                if (this._len === FRAME_SAMPLES) {',
    '                    this.port.postMessage(this._buf.slice(0));',
    '                    this._len = 0;',
    '                }',
    '            }',
    '        }',
    '        return true;',
    '    }',
    '}',
    'registerProcessor("pcm-tap", PcmTap);',
].join('\n');

// An audio context with the tap worklet loaded, and the resampler too when the
// context does not run at the decoder's rate (asked for and refused, or a
// browser that ignores the request).
async function prepareAudioContext(options) {
    const ctx = new AudioContext(options);
    if (ctx.sampleRate !== LIVE_SAMPLE_RATE) {
        await ctx.audioWorklet.addModule('/static/resampler.js');
    }
    const url = URL.createObjectURL(new Blob([PCM_WORKLET], { type: 'application/javascript' }));
    try {
        await ctx.audioWorklet.addModule(url);
    } finally {
        URL.revokeObjectURL(url);
    }
    return ctx;
}

function floatToPcm16(frame) {
    const out = new Int16Array(frame.length);
    for (let i = 0; i < frame.length; i++) {
        const clamped = Math.max(-1, Math.min(1, frame[i]));
        out[i] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
    }
    return out;
}

function peakLevel(frame) {
    let peak = 0;
    for (let i = 0; i < frame.length; i++) {
        const v = Math.abs(frame[i]);
        if (v > peak) peak = v;
    }
    return peak;
}

// Shared by the worklet's normal frames and its stop-time flush, so a
// flushed tail is metered and sent exactly like any other frame.
// The microphone level is watched, never changed: both recognisers normalise
// level themselves, and software gain raises the noise between words as much
// as the speech. What cannot be fixed afterwards is clipping (lost waveform)
// and a level so low it sits in the microphone's own hiss, so those are said.
// The thresholds are the server's (dictation/mic.py), the desktop's too.
const CLIP_PEAK = BOOTSTRAP.mic.clip_peak;
const QUIET_PEAK = BOOTSTRAP.mic.quiet_peak;
const QUIET_AFTER_SEC = BOOTSTRAP.mic.quiet_after_sec;
const mic = { loudest: 0, samples: 0, clipped: 0, warned: false };

function watchLevel(frame, peak) {
    mic.samples += frame.length;
    mic.loudest = Math.max(mic.loudest, peak);
    if (peak >= CLIP_PEAK) mic.clipped++;
    if (mic.warned) return;
    if (mic.clipped >= 3) {
        mic.warned = true;
        showStatus('Microphone is clipping: lower its input level', 'is-warn', 6000);
        devMark('browser', 'microphone clipping', {}, { level: 'warn' });
    } else if (mic.samples > QUIET_AFTER_SEC * LIVE_SAMPLE_RATE && mic.loudest < QUIET_PEAK) {
        mic.warned = true;
        showStatus('Microphone is very quiet: check it is the right one, and its level', 'is-warn', 6000);
        devMark('browser', 'microphone very quiet', { peak: mic.loudest.toFixed(3) }, { level: 'warn' });
    }
}

function handleTapFrame(frame) {
    if (!frame || !frame.length) return;
    const peak = peakLevel(frame);
    setLevel(peak);
    watchLevel(frame, peak);
    const pcm = floatToPcm16(frame).buffer;
    if (pendingPcm) {
        pendingPcm.push(pcm);
        pendingSamples += pcm.byteLength / 2;
        while (pendingSamples > PRE_SOCKET_MAX_SEC * LIVE_SAMPLE_RATE) {
            pendingSamples -= pendingPcm.shift().byteLength / 2;
        }
    } else if (liveSocket && liveSocket.readyState === WebSocket.OPEN) {
        sendPcm(pcm);
    }
}

function sendPcm(pcm) {
    devAudioSent(pcm.byteLength);
    liveSocket.send(pcm);
}

// -- what the radiologist sees ---------------------------------------------

function openDictationRegion() {
    const value = editor.value;
    const start = editor.selectionStart ?? value.length;
    const end = editor.selectionEnd ?? start;
    baseOriginal = value;
    baseBefore = value.slice(0, start);
    baseAfter = value.slice(end);
    // Dictation must not weld onto the words either side of it.
    gapBefore = baseBefore && !/\s$/.test(baseBefore) ? ' ' : '';
    gapAfter = baseAfter && !/^[\s.,;:)]/.test(baseAfter) ? ' ' : '';
    // A selected field travels to the server, which answers a pick-list
    // ("left" into [Right / Left]) by the one rule in template_fields.py.
    const selected = value.slice(start, end);
    const fields = selected.match(FIELD_RE) || [];
    return fields.length === 1 && fields[0] === selected ? selected : null;
}

function renderLiveText() {
    const head = baseBefore + (committedText ? gapBefore + committedText + gapAfter : '');
    const keepScroll = editor.scrollTop;
    editor.value = head + baseAfter;
    editor.setSelectionRange(head.length, head.length);
    if (baseAfter) {
        // Mid-report: stay where the radiologist is looking, and show the
        // words still being decoded under the editor (see #livePreview).
        editor.scrollTop = keepScroll;
        paintPreview('');
        livePreview.textContent = previewText;
        livePreview.hidden = !previewText;
    } else {
        editor.scrollTop = editor.scrollHeight;
        livePreview.hidden = true;
        paintPreview(head);
    }
    // The count is the only thing on screen saying the report is growing while
    // you speak. It is one split of the text, unlike the marks and findings
    // scans, which is why it runs every cycle and they do not.
    updateWordCount();
}

// The dimmed tail is painted by the existing marks underlay rather than a
// second overlay mechanism, and nothing is inserted into the textarea, so an
// export can never contain a provisional word.
function paintPreview(settled) {
    if (!editorMarks) return;
    if (!previewText) { editorMarks.replaceChildren(); return; }
    // The settled half is copied in transparently only to push the preview to
    // the right place on the line; the textarea above paints those same
    // characters for real.
    const out = document.createDocumentFragment();
    out.append(settled + (settled ? ' ' : ''));
    const tail = document.createElement('span');
    tail.className = 'preview';
    tail.textContent = previewText;
    out.append(tail);
    editorMarks.replaceChildren(out);
    syncMarksScroll();
}

//: Kept from the markup so the prompt is written down once, in the HTML.
const EDITOR_PLACEHOLDER = editor.getAttribute('placeholder') || '';

function setRecordingUi(on) {
    dictateBtn.setAttribute('aria-pressed', on ? 'true' : 'false');
    dictateBtn.classList.toggle('is-recording', on);
    micIcon.className = on ? 'fas fa-stop' : 'fas fa-microphone';
    dictateBtn.setAttribute('aria-label', on ? 'Stop recording' : 'Start recording');
    recMeter.hidden = !on;
    cancelBtn.hidden = !on;
    // The live preview is painted in the underlay BEHIND the textarea, and an
    // empty textarea still paints its placeholder on top of it. The two drew
    // over each other as "Predsictherpmicrophone and start speaking." for the
    // first seconds of every dictation, which is the first thing you see.
    editor.placeholder = on ? '' : EDITOR_PLACEHOLDER;
    if (!on) setLevel(0);
}

// The real peak of the last frame: a meter that only ever shows "something"
// is a meter that cannot tell you the microphone is dead.
function setLevel(peak) {
    recMeter.style.setProperty('--level', Math.min(1, peak * 2.2).toFixed(3));
}

function startTimer() {
    recordStartedAt = Date.now();
    const tick = () => {
        const secs = Math.floor((Date.now() - recordStartedAt) / 1000);
        const mm = String(Math.floor(secs / 60)).padStart(2, '0');
        const ss = String(secs % 60).padStart(2, '0');
        recTimer.textContent = `${mm}:${ss}`;
        // The ribbon's "now" edge rides this tick rather than a second timer:
        // it is the same clock, and one clock cannot disagree with itself.
        tlRender();
    };
    tick();
    timerHandle = setInterval(tick, 500);
}

function stopTimer() {
    if (timerHandle) clearInterval(timerHandle);
    timerHandle = null;
}

// -- the session timeline ---------------------------------------------------
// A dictation is a duration, but the status label could only ever say what was
// true this instant and then overwrite itself, so a slow decode and a dead one
// looked identical. This draws the session along time instead: the ribbon is
// the shape of the audio, a hairline inside it is how far the text trails the
// microphone, and the trail says the same in words for anyone who would rather
// read than look. Every number here is one the session already had.

const TL_TRAIL_MAX = 4;        // steps kept on screen; older ones fall off
// The floor stops the ribbon rescaling wildly over the first second. Set to 20
// it did the opposite of its job: a 9-second dictation drew into the left half
// and left the right half blank, which reads as a session that stalled rather
// than one that is short.
const TL_MIN_SPAN_SEC = 10;
const TL_BEHIND_SEC = 2.5;     // past this the text is visibly trailing

const tl = {
    startedAt: 0, steps: [], saved: [], lag: [], stalls: [],
    savedTo: 0, audioSec: 0, behind: 0, words: 0, phase: 'idle',
};

function tlBegin() {
    tl.startedAt = performance.now();
    tl.steps = []; tl.saved = []; tl.lag = []; tl.stalls = [];
    tl.savedTo = 0; tl.audioSec = 0; tl.behind = 0; tl.words = 0;
    tl.phase = 'starting';
    sessionTimeline.hidden = false;
    tlStep('Opening the microphone');
}

function tlElapsed() {
    return tl.startedAt ? (performance.now() - tl.startedAt) / 1000 : 0;
}

//: *tone* colours the duration only: 'good' for a step that beat its budget,
//: 'slow' for one the radiologist waited on.
function tlStep(label, seconds = null, tone = '') {
    tl.steps.push({ at: tlElapsed(), label, seconds, tone });
    if (tl.steps.length > 24) tl.steps.shift();
    tlRender();
}

// Fed from devUpdateArrived, which already computes these for every partial:
// one formula, two readers, so the console and the ribbon can never disagree.
function tlUpdate(msg, stats) {
    tl.audioSec = msg.audioSec || 0;
    tl.behind = stats.behind;
    tl.lag.push({ at: stats.sinceStart / 1000, behind: stats.behind });
    tl.words = stats.words;

    // One block per chunk the server actually froze, read from where the frozen
    // text now ends. Inferring it instead from "the word count grew" drew a
    // block per update -- half a second each, butted end to end -- so the
    // ribbon came out as one flat bar with no boundaries in it, and the
    // boundaries are the whole reason to look at it.
    const saved = msg.savedSec || 0;
    if (saved > tl.savedTo) {
        tl.saved.push({ from: tl.savedTo, to: saved });
        tl.savedTo = saved;
    }

    // The stretch where the live preview was switched off, held open until the
    // state changes back, so it draws as one band rather than a dotted mess.
    const stalling = msg.state === 'catching_up';
    const open = tl.stalls.length && tl.stalls[tl.stalls.length - 1].to === null;
    if (stalling && !open) {
        tl.stalls.push({ from: stats.sinceStart / 1000, to: null });
        tlStep('Catching up');
    } else if (!stalling && open) {
        tl.stalls[tl.stalls.length - 1].to = stats.sinceStart / 1000;
    }

    tl.phase = stalling ? 'behind' : (tl.words ? 'writing' : 'listening');
    tlRender();
}

// What the phase means in plain words. The lag is in here, said as work still
// to do rather than as a number to watch: "about 4s still to write" is the
// same fact as "4s behind" without asking anyone to read it as a metric.
function tlSay() {
    const behindSec = Math.round(tl.behind);
    switch (tl.phase) {
        case 'starting':   return ['Opening the microphone', 'One moment.'];
        case 'listening':  return ['Listening', 'Your first words appear in a few seconds.'];
        // Words, not sentences: the count is a split of the committed text, so
        // saying "sentences" put "12 sentences saved" under a twelve-word line.
        case 'writing':    return ['Writing what you say', tlWords()];
        case 'behind':     return ['Still hearing you',
            `About ${Math.max(1, behindSec)}s of speech still to write. Nothing is lost.`];
        case 'finishing':  return ['Finishing the last sentence', ''];
        case 'polishing':  return ['Yours to edit', 'Still checking the least certain sentences.'];
        case 'kept':       return ['Kept your edits', 'The polish was dropped rather than overwrite you.'];
        case 'done':       return ['Finished', tlWords()];
        default:           return ['Ready', ''];
    }
}

function tlWords() {
    return tl.words ? `${tl.words} word${tl.words === 1 ? '' : 's'} written.` : '';
}

function tlRender() {
    if (sessionTimeline.hidden) return;
    const now = tlElapsed();
    // A little headroom past the live edge, so the "now" line always has room
    // to draw inside the ribbon instead of being clipped against its end, and
    // so the session visibly has somewhere left to run.
    const span = Math.max(TL_MIN_SPAN_SEC, now * 1.06);
    const pct = (s) => Math.max(0, Math.min(100, (s / span) * 100));
    const band = (from, to, cls) =>
        `<span class="tl-seg ${cls}" style="left:${pct(from).toFixed(2)}%;width:${(pct(to) - pct(from)).toFixed(2)}%"></span>`;

    const bands = tl.saved.map((s) => band(s.from, s.to, 'is-saved'));
    // The sentence still being written: from the last saved word up to the
    // last audio the server has read.
    if (tl.audioSec > tl.savedTo) bands.push(band(tl.savedTo, tl.audioSec, 'is-open'));
    for (const s of tl.stalls) bands.push(band(s.from, s.to === null ? now : s.to, 'is-behind'));

    tlRibbon.querySelectorAll('.tl-seg').forEach((n) => n.remove());
    tlRibbon.insertAdjacentHTML('afterbegin', bands.join(''));
    tlNow.style.left = pct(now).toFixed(2) + '%';
    tlDrawPulse(span);

    const [msg, sub] = tlSay();
    tlLiveMsg.textContent = msg;
    tlLiveSub.textContent = sub;

    tlTrail.replaceChildren(...tl.steps.slice(-TL_TRAIL_MAX).map((s) => {
        const li = document.createElement('li');
        const t = document.createElement('span');
        t.className = 'tl-t';
        t.textContent = fmtClock(s.at);
        const label = document.createElement('span');
        label.textContent = s.label;
        const dur = document.createElement('span');
        dur.className = 'tl-dur' + (s.tone ? ` is-${s.tone}` : '');
        dur.textContent = s.seconds === null ? '' : `${s.seconds.toFixed(1)}s`;
        li.append(t, label, dur);
        return li;
    }));
}

function fmtClock(seconds) {
    const s = Math.max(0, Math.floor(seconds));
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

function tlDrawPulse(span) {
    if (!tl.lag.length) { tlPulse.replaceChildren(); return; }
    const W = 1000, H = 40;
    const peak = Math.max(3, ...tl.lag.map((p) => p.behind));
    const d = tl.lag.map((p, i) =>
        `${i ? 'L' : 'M'}${((p.at / span) * W).toFixed(1)} ${(H - 2 - (p.behind / peak) * (H - 8)).toFixed(1)}`
    ).join(' ');
    // Built as markup rather than nodes because it is one path, redrawn whole.
    tlPulse.innerHTML = `<path d="${d}" fill="none" stroke="${
        tl.behind > TL_BEHIND_SEC ? 'var(--warn-text)' : 'var(--glow)'
    }" stroke-width="1.25" stroke-linejoin="round" vector-effect="non-scaling-stroke" opacity=".55" />`;
}

function tlPhase(phase) {
    tl.phase = phase;
    tlRender();
}

// -- the socket -------------------------------------------------------------

function openSocket() {
    const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
    const socket = new WebSocket(`${scheme}://${location.host}/ws/dictate`);
    socket.binaryType = 'arraybuffer';

    socket.onmessage = (event) => {
        const msg = safeParseJson(event.data, null);
        // A previous recording still polishing must never write into the one
        // being dictated now.
        if (!msg || socket !== liveSocket) return;
        if (msg.type === 'partial') {
            tlUpdate(msg, devUpdateArrived(msg));
            committedText = msg.committed || '';
            previewText = msg.preview || '';
            uncertainWords = new Set(msg.uncertain || []);
            renderLiveText();
            showStatus(
                msg.state === 'catching_up'
                    ? 'Catching up'
                    : 'Listening',
                'is-rec',
            );
        } else if (msg.type === 'stopped') {
            devMark('browser', 'report handed back', { words: devWordsIn(msg.text) },
                { ms: stopPressedAt ? performance.now() - stopPressedAt : null });
            // The report is yours now. The accurate re-decode is still running,
            // but you can read and edit while it does.
            previewText = '';
            committedText = msg.text || '';
            uncertainWords = new Set(msg.uncertain || []);
            renderLiveText();
            handedOverText = editor.value;
            learnSnapshot = editor.value;
            finishSession(null);
            // The hand-back is the last word count the trail will quote, so it
            // has to be this text's, not the last partial's.
            tl.words = devWordsIn(msg.text);
            tlStep('Yours to edit',
                stopPressedAt ? (performance.now() - stopPressedAt) / 1000 : null, 'good');
            tlPhase('polishing');
            showStatus('Yours to edit · polishing', 'is-busy');
        } else if (msg.type === 'final') {
            devMark('browser', 'accuracy pass arrived', { words: devWordsIn(msg.text) },
                { ms: stopPressedAt ? performance.now() - stopPressedAt : null });
            const polishSec = stopPressedAt ? (performance.now() - stopPressedAt) / 1000 : null;
            stopPressedAt = 0;
            const improved = msg.text || '';
            if (editor.value !== handedOverText) {
                // You edited while it was working. Your words win: silently
                // replacing them with the machine's would be the worst possible
                // outcome for a clinical report.
                tlStep('Kept your edits', polishSec);
                tlPhase('kept');
                showStatus('Kept your edits', 'is-ok', 4000);
            } else {
                committedText = improved;
                uncertainWords = new Set(msg.uncertain || []);
                renderLiveText();
                learnSnapshot = editor.value;
                pushUndoState();
                announceReportChanged();
                tl.words = devWordsIn(improved);
                tlStep('Checked and polished', polishSec, 'good');
                tlPhase('done');
                showStatus('Polished', 'is-ok', 2000);
            }
            handedOverText = null;
        } else if (msg.type === 'error') {
            devMark('browser', msg.message || 'dictation failed', {}, { level: 'error' });
            showError(msg.message || 'Dictation failed.');
            teardownMic().then(() => finishSession(null));
        }
    };

    // Only meaningful once recording has actually started: startRecording()
    // has its own one-time 'error' listener for the connect attempt itself,
    // and both firing on the same failure raced to set the status text.
    // Only this recording's socket may end it: the previous one closing when its
    // polish finishes used to switch off the microphone mid-dictation.
    const current = () => sessionOpen() && socket === liveSocket;
    socket.onerror = () => { if (current()) showError('Lost the connection to the dictation service.'); };
    // Closed before handing the report back: keep what arrived, and give the
    // editor back rather than leaving it locked to a session that is gone.
    socket.onclose = () => { if (current()) teardownMic().then(() => finishSession(null)); };
    return socket;
}

// A connect to our own loopback server has been observed to error out
// immediately under load, then succeed on the very next attempt with no
// other change: transient, not a broken route. One retry absorbs it instead
// of surfacing a scary error for something that would have worked a moment
// later.
async function connectDictationSocket(attemptsLeft = 2) {
    const socket = openSocket();
    try {
        await new Promise((resolve, reject) => {
            socket.addEventListener('open', resolve, { once: true });
            socket.addEventListener('error', reject, { once: true });
        });
        return socket;
    } catch {
        socket.close();
        if (attemptsLeft > 1) return connectDictationSocket(attemptsLeft - 1);
        throw new Error('Could not reach the dictation service. Check that it is running and try again.');
    }
}

// One undo entry per dictation and one scan of the finished report: not one of
// each per partial, which would both blow the 50-entry history in seconds and
// spend the whole machine re-scanning half-sentences.
function finishSession(okMessage) {
    setSessionPhase('idle');
    stopTimer();
    setRecordingUi(false);
    previewText = '';
    paintPreview('');
    livePreview.hidden = true;
    pushUndoState();
    announceReportChanged();
    syncReportButtons();
    if (okMessage) showStatus(okMessage, 'is-ok', 2500);
}

// A partial frame (under 320 samples) can be sitting in the worklet's
// accumulator when recording stops; disconnecting drops it on the floor
// before it ever reaches the socket. Ask the worklet for whatever it is
// still holding and let it through the normal frame handler before tearing
// the node down, so the last fraction of a second of speech is not stranded.
// The reply carries a `flushed` field so an ordinary frame posted in the
// meantime cannot be mistaken for it: waiting on whatever arrives first would
// let the tail be dropped by the very disconnect this exists to delay.
function flushMicTap(node) {
    return new Promise((resolve) => {
        const port = node.port;
        // Exactly one of the two paths may ever run. Without this, a late
        // worklet reply after the timeout still pushed a PCM frame down the
        // socket behind the stop message, on a node already disconnected.
        const timeout = setTimeout(() => {
            port.onmessage = null;
            resolve();
        }, MIC_FLUSH_TIMEOUT_MS);
        port.onmessage = (event) => {
            const data = event.data;
            if (data && data.flushed) {
                clearTimeout(timeout);
                port.onmessage = null;
                handleTapFrame(data.flushed);
                resolve();
                return;
            }
            handleTapFrame(data);
        };
        port.postMessage({ command: 'flush' });
    });
}

async function teardownMic() {
    isRecording = false;
    pendingPcm = null;  // a failed start must not leak its held sound into the next recording
    if (micNode) {
        // Take the node out of the global BEFORE awaiting, so a Stop and a
        // Cancel milliseconds apart cannot both reach the same node: the
        // second call sees null and does nothing, instead of throwing part
        // way through and leaving the report and the buttons mid-recording.
        const node = micNode;
        micNode = null;
        await flushMicTap(node);
        node.disconnect();
    }
    if (micStream) { micStream.getTracks().forEach((t) => t.stop()); micStream = null; }
    if (audioContext) {
        await audioContext.close().catch(() => {});
        audioContext = null;
    }
}

// -- start / stop / cancel --------------------------------------------------

// A stream to record instead of the microphone, once: set by the developer
// Test voice (speechtest.js), which the app runs without.
let substituteMic = null;

async function startRecording() {
    if (sessionOpen()) return;
    setSessionPhase('starting');
    devRecordingStarted();
    // Before this, the button, the timer and the status all stayed idle while
    // the microphone permission, the audio graph and the socket were awaited:
    // press record, and for a second or more nothing on screen moved at all.
    tlBegin();
    showStatus('Opening the microphone', 'is-busy');
    // The audio graph comes first: an open microphone delivers sound at once,
    // and anything it delivers before the graph exists has nowhere to go.
    try {
        // Asking the context for 16 kHz makes the browser resample for us, so
        // there is no hand-written downsampler to get wrong.
        audioContext = await prepareAudioContext({ sampleRate: LIVE_SAMPLE_RATE });
    } catch (err) {
        console.error('Could not start live dictation:', err);
        await teardownMic();
        setSessionPhase('idle');
        showError('Could not start dictation: ' + (err.message || 'unknown error'));
        return;
    }
    const micAskedAt = performance.now();
    try {
        // The browser's own voice-call processing is switched off. Noise
        // suppression and automatic gain both alter speech in ways recognisers
        // were not trained on (Google's and Deepgram's guidance for
        // transcription: send unaltered audio), and gain control also drifts
        // the microphone level during a session. Echo cancellation has nothing
        // to cancel: this page plays no sound.
        micStream = substituteMic || await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: false,
            },
        });
        substituteMic = null;
    } catch (err) {
        devMark('browser', 'microphone refused', {}, { level: 'error' });
        console.error('Microphone access denied:', err);
        showError('Microphone access denied. Allow microphone permissions in your browser settings, then try again.');
        await teardownMic();  // closes the audio graph built above
        setSessionPhase('idle');
        tlStep('Microphone refused');
        sessionTimeline.hidden = true;
        return;
    }

    const applied = micStream.getAudioTracks()[0]?.getSettings?.() || {};
    devMark('browser', 'microphone granted', {
        rate: applied.sampleRate, ns: applied.noiseSuppression,
        agc: applied.autoGainControl, ec: applied.echoCancellation,
    }, { ms: performance.now() - micAskedAt });
    Object.assign(mic, { loudest: 0, samples: 0, clipped: 0, warned: false });
    tlStep('Microphone ready', (performance.now() - micAskedAt) / 1000);

    try {
        // Listen before connecting: the first words are often spoken while the
        // socket is still opening, and they are held, not thrown away.
        pendingPcm = [];
        pendingSamples = 0;
        let source;
        try {
            source = audioContext.createMediaStreamSource(micStream);
        } catch (err) {
            // Firefox: a microphone can only feed a context at its own rate.
            // Run at that rate and resample in the worklet instead.
            devMark('browser', 'resampling in the page', { reason: err.name });
            await audioContext.close();
            audioContext = await prepareAudioContext({});
            source = audioContext.createMediaStreamSource(micStream);
        }
        micNode = new AudioWorkletNode(audioContext, 'pcm-tap', {
            processorOptions: { inRate: audioContext.sampleRate, outRate: LIVE_SAMPLE_RATE },
        });
        micNode.port.onmessage = (event) => handleTapFrame(event.data);
        source.connect(micNode);

        const socketAskedAt = performance.now();
        liveSocket = null;  // the previous recording's socket stops being heard from here
        liveSocket = await connectDictationSocket();
        devMark('browser', 'dictation socket open', {}, { ms: performance.now() - socketAskedAt });
        tlStep('Connected', (performance.now() - socketAskedAt) / 1000);

        const field = openDictationRegion();
        if (field) liveSocket.send(JSON.stringify({ command: 'field', text: field }));

        const held = pendingPcm;
        pendingPcm = null;
        held.forEach(sendPcm);
    } catch (err) {
        console.error('Could not start live dictation:', err);
        await teardownMic();
        setSessionPhase('idle');
        showError('Could not start dictation: ' + (err.message || 'unknown error'));
        return;
    }

    flushLearning();
    committedText = '';
    previewText = '';
    // The doubts belong to the dictation that raised them. A new session will
    // report its own, and text from the last one has already been read.
    uncertainWords = new Set();
    isRecording = true;
    setSessionPhase('recording');
    setRecordingUi(true);
    startTimer();
    devMark('browser', 'recording started', {}, { ms: performance.now() - dev.recordStart });
    tlStep('Listening');
    tlPhase('listening');
    showStatus('Listening', 'is-rec');
}

async function stopRecording() {
    if (sessionPhase !== 'recording') return;
    // Owned until the hand-back ('stopped') arrives: it carries the last
    // section, which a recording started meanwhile would have thrown away.
    setSessionPhase('finishing');
    stopPressedAt = performance.now();
    devMark('browser', 'stop pressed', { updates: dev.updates });
    await teardownMic();
    setRecordingUi(false);
    stopTimer();
    tlStep('Stopped');
    tlPhase('finishing');
    showStatus('Finishing', 'is-busy');
    if (liveSocket && liveSocket.readyState === WebSocket.OPEN) {
        liveSocket.send(JSON.stringify({ command: 'stop' }));
    } else {
        finishSession(null);
    }
}

// Cancel means cancel: the report goes back to exactly what it was.
async function cancelRecording() {
    devMark('browser', 'recording discarded', { updates: dev.updates });
    if (liveSocket && liveSocket.readyState === WebSocket.OPEN) {
        liveSocket.send(JSON.stringify({ command: 'cancel' }));
    }
    await teardownMic();
    committedText = '';
    previewText = '';
    editor.value = baseOriginal;
    paintPreview('');
    livePreview.hidden = true;
    stopTimer();
    setRecordingUi(false);
    setSessionPhase('idle');
    // Discard means discard: the session it described is gone, so its timeline
    // goes with it rather than sitting under a report it no longer describes.
    sessionTimeline.hidden = true;
    tl.phase = 'idle';
    hideStatus();
}

dictateBtn.addEventListener('click', () => {
    if (sessionPhase === 'recording') stopRecording();
    else if (sessionPhase === 'idle') startRecording();
});
cancelBtn.addEventListener('click', cancelRecording);
