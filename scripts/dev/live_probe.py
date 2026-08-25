"""Measure real live-dictation latency over /ws/dictate — no browser needed.

Streams a wav file to the socket at real-time pace (as if spoken live) and
logs, for every server update, the wall-clock lag behind the audio position
it reflects. That lag is exactly what a radiologist watching the screen
would feel as "words appear slowly".

Usage:
    python -m scripts.dev.live_probe data/bench_audio/chest_short.wav
    python -m scripts.dev.live_probe data/bench_audio/chest_short.wav --url ws://127.0.0.1:8005/ws/dictate
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
import wave

import numpy as np
from scipy.signal import resample_poly
import websockets

LIVE_SR = 16000
FRAME_MS = 100  # how often the browser's AudioWorklet posts a chunk


def load_pcm16(path: str) -> bytes:
    with wave.open(path, "rb") as w:
        raw = w.readframes(w.getnframes())
        sr = w.getframerate()
        sampwidth = w.getsampwidth()
        channels = w.getnchannels()

    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if sr != LIVE_SR:
        audio = resample_poly(audio, LIVE_SR, sr).astype(np.float32)
    pcm16 = np.clip(audio * 32768.0, -32768, 32767).astype("<i2")
    return pcm16.tobytes()


async def run(path: str, url: str) -> None:
    pcm = load_pcm16(path)
    bytes_per_frame = int(LIVE_SR * FRAME_MS / 1000) * 2
    total_sec = len(pcm) / 2 / LIVE_SR
    print(f"streaming {total_sec:.1f}s of audio, {len(pcm) // bytes_per_frame} frames")

    async with websockets.connect(url, max_size=None) as ws:
        t0 = time.time()

        async def sender():
            for i in range(0, len(pcm), bytes_per_frame):
                await ws.send(pcm[i:i + bytes_per_frame])
                await asyncio.sleep(FRAME_MS / 1000)
            await ws.send(json.dumps({"command": "stop"}))

        async def receiver():
            first_word_at = None
            async for raw in ws:
                t = time.time() - t0
                msg = json.loads(raw)
                if msg["type"] == "partial":
                    committed_len = len(msg["committed"])
                    preview_len = len(msg.get("preview") or "")
                    if first_word_at is None and (committed_len or preview_len):
                        first_word_at = t
                        print(f"[{t:6.2f}s] FIRST TEXT APPEARS  audioSec={msg['audioSec']}")
                    print(f"[{t:6.2f}s] partial  audioSec={msg['audioSec']:>5}  "
                          f"lag={t - msg['audioSec']:+.2f}s  committed_len={committed_len} preview_len={preview_len}")
                elif msg["type"] == "stopped":
                    print(f"[{t:6.2f}s] stopped  -> committed text handed back immediately")
                    print(f"           text: {msg['text']!r}")
                elif msg["type"] == "final":
                    print(f"[{t:6.2f}s] final    -> polish done, {t - total_sec:.2f}s after audio ended")
                    print(f"           text: {msg['text']!r}")
                elif msg["type"] == "error":
                    print(f"[{t:6.2f}s] ERROR: {msg['message']}")

        await asyncio.gather(sender(), receiver())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--url", default="ws://127.0.0.1:8005/ws/dictate")
    args = ap.parse_args()
    asyncio.run(run(args.wav, args.url))
