# CODING_STANDARDS.md

These are the conventions already in force across `src/`. Descriptive, not
aspirational — match them so the codebase stays uniform. For architecture and
module boundaries, and for `.claude/` agents/skills/workflows, see
[CLAUDE.md](CLAUDE.md).

## Language & typing

- Start every module with `from __future__ import annotations`.
- Type-hint all public functions (params and return). Prefer precise types
  (`Optional[str]`, `List[dict]`, `Path`) over bare containers.
- `snake_case` for functions/variables/modules, `PascalCase` for classes,
  `_UPPER` module-level constants. Single leading underscore for non-public
  helpers and attributes.

## Module shape

- Open each module with a docstring stating its **single responsibility** (and,
  where useful, the lifecycle/data-flow it participates in). If you can't name
  the responsibility in one line, the module is doing too much.
- One responsibility per file; sequence/orchestrate in a dedicated file (e.g.
  `postprocess/pipeline.py`, `cloud/sync_manager.py`) rather than letting layers
  reach into each other.

## Imports

- Standard top-of-file imports for lightweight, always-present dependencies.
- **Lazily import heavy or optional dependencies inside the function that uses
  them** — PySide6, `httpx`, `soundfile`, `keyring`, `rapidfuzz`, and the cloud
  modules are imported at call sites, not module top-level. Keeps the offline
  pipeline importable without cloud/GUI deps and keeps startup fast.
- No unused imports (`ruff` F401 must be clean).

## Errors & logging

- Each module has `logger = logging.getLogger(__name__)`.
- **Never swallow an exception silently.** A best-effort/graceful-degradation
  block must at least `logger.debug(...)` or `logger.warning(...)` the cause
  before returning a fallback. Reserve broad `except Exception` for genuine
  degrade-don't-crash paths and log them.
- Raise domain exceptions from the `CloudError` hierarchy
  (`src/cloud/exceptions.py`) for cloud failures so callers can catch the whole
  subsystem in one clause.

## Security & privacy

- **No hardcoded secrets.** API keys/tokens live in the OS keychain via
  `keyring`; the project id (non-secret) lives in settings. Nothing secret is
  logged.
- Any data destined to leave the device passes through `DeIdentifier` and
  `validate_clean()` first; on failure the record is dropped.
- Treat downloaded/external archives as untrusted — validate member paths
  before extracting (see `cloud/framework/sync.py::_extract_model`).

## Patient-safety (clinical ML output)

- **Abstain by default.** A model-derived clinical suggestion surfaces only
  when it clears a calibrated confidence gate (`imaging/abstention.py`); below
  it, withhold rather than guess. Defaults err toward silence.
- **Show the evidence.** Any surfaced imaging finding must carry a region the
  user can inspect; a confident-but-unexplained finding is withheld.
- **Label it as assistive.** Every clinical-ML result carries a non-diagnostic
  disclaimer; the radiologist's report is authoritative.
- **Never regress silently.** A training run that doesn't beat its validated
  baseline on a held-out split is rejected, not shipped.

## File I/O & state

- Route filesystem paths through `features/file_manager.py` helpers rather
  than hardcoding paths; this keeps `data/` layout in one place.
- Persistent state is file-first and human-inspectable (JSON settings/registry,
  SQLite only where relational queries are needed, e.g. the staging DB).

## Tests

- `pytest` under `tests/`, mirroring module names (`test_<module>.py`).
- Cover safety-critical paths explicitly: PHI de-identification, the consent
  gate, and untrusted-archive extraction. Use `tmp_path` and mock cloud
  clients — tests must not touch the network.
- `python -m pytest tests/ -q` and `ruff check src tests` must both be green
  before shipping a change.