"""Dictation pipeline: audio capture → Whisper → 10-stage post-processing.

This package owns the full speech-to-corrected-text path. UI, settings,
and domain knowledge (medical dictionary, macros, critical findings)
live elsewhere and call into this package.

Public surface (see submodules for details):
    audio.Recorder              microphone → growing WAV
    transcriber.Transcriber     faster-whisper wrapper
    worker.LiveTranscribeWorker QThread sliding-window worker
    postprocess.postprocess_transcript  correction pipeline entry point
"""
