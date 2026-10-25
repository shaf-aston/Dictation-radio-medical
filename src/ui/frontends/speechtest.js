// ---------------------------------------------------------------------------
// Test voice (developer tool, src/devtools/speech_test.py). The server speaks
// the typed text; the WAV is played into the ordinary recording path as if it
// were the microphone (app.js's substituteMic), and Stop is pressed when it
// ends. Loaded after app.js, whose functions it calls.
// ---------------------------------------------------------------------------

document.getElementById('speechTestForm').addEventListener('submit', async (event) => {
    event.preventDefault();
    const text = document.getElementById('speechTestText').value.trim();
    const btn = document.getElementById('speechTestBtn');
    if (!text || sessionOpen()) return;
    btn.disabled = true;
    try {
        const response = await fetch('/api/dev/speak', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
        });
        if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
        const audio = new Audio(URL.createObjectURL(await response.blob()));
        // Play for an instant so the stream has a track, then rewind and wait
        // until recording is live: otherwise the opening words are lost.
        await audio.play();
        audio.pause();
        audio.currentTime = 0;
        substituteMic = audio.captureStream();
        audio.addEventListener('ended', () => {
            // A second of silence first, so the last words are not cut off.
            setTimeout(() => { if (isRecording) stopRecording(); }, 1000);
        }, { once: true });
        devMark('browser', 'test voice playing', { sec: Math.round(audio.duration) });
        await startRecording();
        if (isRecording) await audio.play();
    } catch (err) {
        showError('Test voice failed: ' + err.message);
    } finally {
        btn.disabled = false;
    }
});
