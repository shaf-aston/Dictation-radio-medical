import logging
import threading
from typing import Any, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

logger = logging.getLogger(__name__)

_CLIP_THRESHOLD = 0.95   # fraction of full scale; above this = clipping


class Recorder:
    def __init__(self, samplerate: int = 16000, channels: int = 1):
        self.samplerate = samplerate
        self.channels = channels
        self._stream: Optional[Any] = None
        self._sf: Optional[sf.SoundFile] = None
        self._lock = threading.Lock()
        self._recording = False
        self._current_level: float = 0.0   # RMS amplitude 0.0–1.0
        self._is_clipping: bool = False

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def current_level(self) -> float:
        """RMS amplitude of the most recent audio block, normalised 0.0–1.0."""
        return self._current_level

    @property
    def is_clipping(self) -> bool:
        """True when the most recent block contained samples near full scale."""
        return self._is_clipping

    def start(self, file_path: str) -> None:
        if self._recording:
            return
        self._current_level = 0.0
        self._is_clipping = False
        self._sf = sf.SoundFile(
            file_path,
            mode="w",
            samplerate=self.samplerate,
            channels=self.channels,
            subtype="PCM_16",
        )

        def callback(indata, frames, time, status):
            if status:
                logger.debug("Audio stream status: %s", status)
            with self._lock:
                if self._sf is not None:
                    self._sf.write(indata.copy())
            # Compute RMS level from int16 samples normalised to [-1, 1]
            float_data = indata.astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(float_data ** 2)))
            self._current_level = min(rms * 8.0, 1.0)   # scale: 0.125 RMS ≈ full bar
            self._is_clipping = bool(np.any(np.abs(float_data) >= _CLIP_THRESHOLD))

        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="int16",
            callback=callback,
            blocksize=1024,       # ~64ms chunks at 16kHz – low latency, stable writes
            latency="low",        # request low-latency from the audio driver
        )
        self._stream.start()
        self._recording = True

    def stop(self) -> None:
        if not self._recording:
            return
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        finally:
            self._stream = None
            with self._lock:
                if self._sf is not None:
                    self._sf.close()
                    self._sf = None
            self._recording = False
