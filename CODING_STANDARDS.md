# Coding Standards — Radio Dictate

Consistent style makes the codebase easy to navigate and extend.
These rules apply to all Python files under `src/`.

---

## 1. Naming

| Thing | Convention | Example |
|-------|-----------|---------|
| Module-level constant | `UPPER_SNAKE_CASE` | `_WINDOW_SEC = 25.0` |
| Public constant | `UPPER_SNAKE_CASE` | `SUPPORTED_MODELS` |
| Private constant | `_UPPER_SNAKE_CASE` | `_HALLUCINATION_PHRASES` |
| Class | `PascalCase` | `LiveTranscribeWorker` |
| Function / method | `lower_snake_case` | `scan_for_critical_findings` |
| Variable | `lower_snake_case` | `committed_text`, `chunk_start_sec` |
| Private method | `_lower_snake_case` | `_ensure_model`, `_build_output` |
| Boolean variable | verb prefix | `is_recording`, `vad_enabled`, `has_committed` |

**Be specific.** Avoid `data`, `tmp`, `result`, `x`. Name variables after what
they contain: `patient_id`, `segment_end_sec`, `correction_count`.

---

## 2. Type annotations

Annotate every function signature. Return type is mandatory.

```python
# Good
def scan_for_critical_findings(text: str) -> List[CriticalFinding]:
    ...

def _write(action: str, detail: Dict[str, Any]) -> None:
    ...

# Bad — unannotated
def scan(text):
    ...
```

Use `Optional[X]` for values that can be `None`. Use `List`, `Dict`, `Tuple`
from `typing` (Python 3.9-compatible syntax). For internal-only helpers where
the type is obvious from context, annotations may be omitted on `self`/`cls`.

---

## 3. Docstrings

Every **public function, method, and class** must have a docstring.
Private helpers (`_name`) need one only if the logic is non-obvious.

**Format:**
```python
def autosave_report(text: str, patient_info: dict) -> Optional[str]:
    """Write report to the autosave folder.

    Skips silently if text is empty.

    Args:
        text: Report body text (plain text, not HTML).
        patient_info: Dict with keys 'id', 'name', 'dob', etc.

    Returns:
        Absolute path of the saved file, or None on error.
    """
```

One-line docstrings are fine for simple functions:
```python
def cleanup_temp_files() -> None:
    """Delete all files in the temp/ directory."""
```

Do **not** write docstrings that just repeat the function name:
```python
# Bad
def stop(self) -> None:
    """Stop the worker."""  # adds nothing
```

---

## 4. Inline comments

Comments explain **why**, not what. The code says what.

```python
# Good — explains a non-obvious constraint
beam_size = 1  # beam=1 gives 3× speedup with minimal accuracy loss at live latency

# Bad — restates the code
beam_size = 1  # set beam size to 1
```

One comment per logical block is usually enough. No multi-line comment walls.
No `# TODO: fix this` without a concrete description of the problem.

---

## 5. Imports

**Order** (PEP 8): stdlib → third-party → local (`src.*`). Separated by blank lines.

```python
import json
import logging
from typing import Dict, List

import numpy as np
from PySide6.QtCore import QObject, Signal

from src.dictation.transcriber import Transcriber
from src.features.file_manager import temp_dir
```

**No inline imports** inside functions, except for lazy-loading heavy
dependencies to defer their import cost:

```python
# Acceptable — defers ctranslate2 import until first transcription
def _ensure_model(self) -> None:
    from faster_whisper import WhisperModel
    ...
```

**Always absolute** from `src`:
```python
from src.dictation.transcriber import Transcriber   # correct
from transcriber import Transcriber            # broken outside src/
```

---

## 6. Logging

Use the module-level logger everywhere. No `print()` in production code.

```python
# Top of every module
logger = logging.getLogger(__name__)

# Usage
logger.debug("Cycle %d: chunk=%.1fs elapsed=%.2fs", cycle, chunk_sec, elapsed)
logger.info("Model loaded: %s with compute_type=%s", model_size, ct)
logger.warning("Auto-save failed: %s", exc)
logger.error("Worker crashed: %s", exc, exc_info=True)
```

Use `%`-style formatting in logger calls (not f-strings) — the string is only
formatted if that log level is active.

---

## 7. Error handling

Handle errors at system boundaries (user input, file I/O, hardware). Let
internal errors propagate so they surface clearly during development.

```python
# Good — boundary error handling with informative log
try:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
except OSError as exc:
    logger.warning("Save failed: %s", exc)
    return None

# Bad — swallows all errors silently
try:
    do_something()
except:
    pass
```

Do not add `try/except` around code that cannot raise. Trust internal module
contracts.

---

## 8. Constants

Define tuning constants at module level, not buried in functions or methods.
Group them with a section comment.

```python
# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
_WINDOW_SEC = 25.0       # max audio duration per transcription cycle
_OVERLAP_SEC = 3.0       # context overlap when sliding window forward
_MIN_AUDIO_SEC = 0.8     # ignore recordings shorter than this
```

Add a brief comment on the right for any constant whose purpose is not
immediately obvious from the name.

---

## 9. Classes

```python
class Transcriber:
    """Wraps faster-whisper for radiology transcription.

    Lazy-loads the model on first call to transcribe() to avoid startup
    delay. Applies the radiology initial prompt unless disabled.

    Args:
        model_size: One of SUPPORTED_MODELS (default 'base').
        device: 'auto', 'cpu', or 'cuda'.
        compute_type: ctranslate2 quantisation type; auto-selected if None.
        use_msk_prompt: Whether to inject the radiology domain prompt.
    """
```

- Always annotate `__init__` parameters.
- Group methods logically with section comments (`# --- Public control ---`).
- Keep `__init__` to assignment only; defer expensive work to a `load()` or
  `_ensure_X()` method so the object constructs instantly.

---

## 10. File structure template

```python
"""
One-line module summary.

Longer description if needed — explain the module's role in the system,
key design decisions, or any gotchas.
"""

# --- stdlib imports ---
import logging
from typing import List

# --- third-party imports ---
import numpy as np

# --- local imports ---
from src.dictation.transcriber import Transcriber

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_SOME_CONSTANT = 42


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def public_function(arg: str) -> str:
    """One-line summary.

    Returns:
        Processed string.
    """
    ...


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _helper(x: int) -> bool:
    ...
```

---

## 11. What NOT to do

- No `print()` — use `logger`
- No bare `except:` — always name the exception
- No `global` state except singletons with explicit thread safety
- No circular imports — check module dependencies before adding an import
- No hardcoded paths — use the path functions in `src/features/file_manager.py`
- No `from module import *` — always name what you import
- No redundant comments that restate the code
