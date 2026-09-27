"""The desktop worker is the web loop (LiveSession) fed from a growing WAV.

It used to be a second copy of that loop and drifted from it. Driven here off a
real file being written by soundfile at speaking pace, the way the recorder
writes it, with a zero-cost fake engine whose words are encoded in the audio.
"""

import threading
import time

import pytest
import soundfile as sf

pytest.importorskip("PySide6")

from scripts.eval import simulate_lag as sim  # noqa: E402
from src.dictation import worker as worker_mod  # noqa: E402
from src.dictation.asr.types import CostModel  # noqa: E402
from src.dictation.stream.live_session import LiveSession  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402

# No Qt event loop runs here, so a queued signal would never arrive: listen
# directly, in the emitting (worker) thread.
DIRECT = Qt.ConnectionType.DirectConnection

SR = 16000


@pytest.fixture(autouse=True)
def _fake_vad(monkeypatch):
    from src.dictation.stream import live_session, polish

    monkeypatch.setattr(live_session, "detect_speech", sim.energy_vad)
    monkeypatch.setattr(polish, "detect_speech", sim.energy_vad)
    monkeypatch.setattr(worker_mod, "build_context_prompt", lambda: "")


def _engine():
    return sim.SimEngine(CostModel(fixed_sec=0.0), serial=False)


def _words(text):
    return sim._seen(text)


def test_desktop_worker_matches_the_web_loop(tmp_path, monkeypatch):
    audio, spoken = sim.build_dictation(12.0, seed=3)
    engine = _engine()
    monkeypatch.setattr(worker_mod, "create_engine", lambda **_kw: engine)

    path = str(tmp_path / "rec.wav")
    writer = sf.SoundFile(path, mode="w", samplerate=SR, channels=1, subtype="PCM_16")
    worker = worker_mod.LiveTranscribeWorker(path, "tiny.en", "en", vad_enabled=False)
    partials, segments, done = [], [], threading.Event()
    worker.partial.connect(lambda text, n: partials.append((text, n)), DIRECT)
    worker.segments.connect(lambda segs: segments.append(segs), DIRECT)
    worker.finished.connect(done.set, DIRECT)

    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()
    block = SR // 10
    for i in range(0, len(audio), block):
        writer.write(audio[i : i + block])
        writer.flush()
        time.sleep(0.02)  # 5x speaking pace
    writer.close()
    time.sleep(0.5)
    worker.finalize()
    assert done.wait(20), "worker never finished"

    # Words appeared while recording, not only at the end.
    assert len(partials) > 2
    # Every committed length points inside its own text.
    assert all(0 <= n <= len(text) for text, n in partials)
    # The final text has every spoken word, once, in order.
    desktop = _words(partials[-1][0])
    assert desktop == [w.index for w in spoken]
    assert segments, "the training collector got no segment timings"

    # The web loop over the same audio says the same words.
    web = LiveSession(_engine(), _engine(), postprocess=False)
    for i in range(0, len(audio), block):
        web.feed(audio[i : i + block])
        web.cycle()
    web.close_open_tail_fast()
    assert _words(web.finalize()) == desktop


def test_desktop_partials_are_raw_text(tmp_path, monkeypatch):
    # The desktop post-processes on its own thread; the worker must not.
    audio, _ = sim.build_dictation(6.0, seed=5)
    engine = _engine()
    monkeypatch.setattr(worker_mod, "create_engine", lambda **_kw: engine)
    path = str(tmp_path / "rec.wav")
    sf.write(path, audio, SR, subtype="PCM_16")
    worker = worker_mod.LiveTranscribeWorker(path, "tiny.en", "en", vad_enabled=False)
    partials, done = [], threading.Event()
    worker.partial.connect(lambda text, n: partials.append(text), DIRECT)
    worker.finished.connect(done.set, DIRECT)
    worker.finalize()
    threading.Thread(target=worker.run, daemon=True).start()
    assert done.wait(20)
    # Raw engine output is lower-case "wN" tokens with no added punctuation.
    assert partials and partials[-1] == partials[-1].lower()
    assert "." not in partials[-1]
