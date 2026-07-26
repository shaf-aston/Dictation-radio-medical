from src.dictation.asr.factory import DEFAULT_ENGINE, create_engine
from src.dictation.asr.port import AsrEngine
from src.dictation.asr.types import AsrResult, AsrSegment, EngineCaps, TranscribeContext, Word
# The radiology priming vocabulary is the same for every engine, so it belongs
# to this seam rather than to one engine's wrapper. Re-exported here so nothing
# outside asr/ has to import transcriber.py (the faster-whisper detail) to get it.
from src.dictation.transcriber import RADIOLOGY_PROMPT

__all__ = [
    "AsrEngine",
    "AsrResult",
    "AsrSegment",
    "EngineCaps",
    "TranscribeContext",
    "Word",
    "create_engine",
    "DEFAULT_ENGINE",
    "RADIOLOGY_PROMPT",
]
