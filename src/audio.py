import threading
import sounddevice as sd
import soundfile as sf


class Recorder:
    def __init__(self, samplerate: int = 16000, channels: int = 1):
        self.samplerate = samplerate
        self.channels = channels
        self._stream = None
        self._sf = None
        self._lock = threading.Lock()
        self._recording = False

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start(self, file_path: str) -> None:
        if self._recording:
            return
        self._sf = sf.SoundFile(file_path, mode="w", samplerate=self.samplerate, channels=self.channels, subtype="PCM_16")

        def callback(indata, frames, time, status):
            if status:
                # Dropouts or overflows are possible on low-spec machines
                pass
            with self._lock:
                if self._sf is not None:
                    self._sf.write(indata.copy())

        self._stream = sd.InputStream(samplerate=self.samplerate, channels=self.channels, dtype="int16", callback=callback)
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
