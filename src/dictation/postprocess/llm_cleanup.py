"""Optional LLM cleanup of a dictated report via Groq.

A consent-gated, on-demand polish for a *finished* report — fixing grammar,
run-on sentences, and obvious transcription slips that the rule-based pipeline
cannot. It is deliberately NOT part of the per-chunk live pipeline: it costs an
API call and adds latency, so the user triggers it explicitly on the full
document ("AI Cleanup"), and it is off by default.

Safety / privacy:
  * Off unless ``groq_cleanup_enabled`` AND ``cloud_training_consent`` are set —
    the same consent the cloud-training path uses (no extra surprise network).
  * Text is run through :class:`~src.medical.deid.DeIdentifier` BEFORE it leaves
    the device, so patient identifiers are not sent to Groq.
  * Degrade-don't-crash: any error (missing key, network, bad JSON) returns the
    input unchanged, logged. Cleanup must never lose the radiologist's text.
  * The model is instructed to correct only language/transcription errors and to
    never invent or remove clinical content.

Secret handling mirrors the Lightning client: the Groq API key lives in the OS
keychain, never in settings or logs.
"""

from __future__ import annotations

import json
import logging
from typing import Any, List, Optional, Tuple, cast

from src.core.keychain import clear_secret, get_secret, store_secret

logger = logging.getLogger(__name__)

_KEYRING_KEY = "groq_api_key"
_DEFAULT_MODEL = "llama-3.3-70b-versatile"
_TIMEOUT = 30.0

_SYSTEM_PROMPT = (
    "You are a careful copy-editor for radiology dictation. Fix only grammar, "
    "punctuation, capitalisation, spacing, and obvious speech-to-text errors. "
    "NEVER add, remove, or change clinical findings, measurements, laterality, "
    "or negations. If unsure, leave the text unchanged. Preserve the report's "
    "structure and headings. Return strict JSON."
)

_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "cleanup",
        "schema": {
            "type": "object",
            "properties": {
                "corrected_text": {"type": "string"},
                "changes": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["corrected_text"],
            "additionalProperties": False,
        },
    },
}


# ---------------------------------------------------------------------------
# Key handling
# ---------------------------------------------------------------------------

def store_groq_key(api_key: str) -> None:
    """Persist the Groq API key in the OS keychain."""
    store_secret(_KEYRING_KEY, api_key)
    logger.info("Groq API key stored in OS keychain")


def get_groq_key() -> Optional[str]:
    """Retrieve the Groq API key from the OS keychain, or None if unset."""
    return get_secret(_KEYRING_KEY)


def clear_groq_key() -> None:
    """Remove the stored Groq API key from the OS keychain."""
    clear_secret(_KEYRING_KEY)


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

def is_enabled() -> bool:
    """True only if the user enabled Groq cleanup AND granted cloud consent."""
    try:
        from src.core.settings import Settings
        s = Settings()
        return bool(s.get("groq_cleanup_enabled")) and bool(s.get("cloud_training_consent"))
    except Exception as exc:
        logger.debug("Could not read Groq cleanup settings; disabled: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def clean_with_llm(
    text: str, patient_info: Optional[dict] = None
) -> Tuple[str, List[str]]:
    """Polish *text* via Groq; return ``(cleaned_text, changes)``.

    Returns the input unchanged (with an empty change list) if cleanup is
    disabled, the key is missing, or anything fails — never raises. ``patient_info``
    (if given) is used to de-identify the text before it is sent.

    Args:
        text: The finished report text to polish.
        patient_info: Optional known identifiers to scrub before upload.

    Returns:
        ``(cleaned_text, changes)`` where ``changes`` is a short human-readable
        list of edits the model reported.
    """
    if not text.strip():
        return text, []
    if not is_enabled():
        logger.debug("Groq cleanup disabled; returning text unchanged")
        return text, []

    api_key = get_groq_key()
    if not api_key:
        logger.warning("Groq cleanup enabled but no API key configured")
        return text, []

    # PHI never leaves the device: scrub before sending.
    from src.medical.deid import DeIdentifier
    deid = DeIdentifier(patient_info or {})
    safe_text = deid.deidentify_text(text)

    try:
        from src.core.settings import Settings
        model = Settings().get("groq_model", _DEFAULT_MODEL) or _DEFAULT_MODEL
    except Exception:
        model = _DEFAULT_MODEL

    try:
        from groq import Groq
    except ImportError:
        logger.warning("Groq package not installed; skipping cleanup")
        return text, []

    try:
        client = Groq(api_key=api_key, timeout=_TIMEOUT)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": safe_text},
            ],
            response_format=cast("Any", _RESPONSE_SCHEMA),
            temperature=0.0,
        )
        content = resp.choices[0].message.content
        payload = json.loads(content) if content else {}
    except Exception as exc:
        logger.warning("Groq cleanup failed; returning text unchanged: %s", exc)
        return text, []

    cleaned = payload.get("corrected_text")
    if not isinstance(cleaned, str) or not cleaned.strip():
        logger.warning("Groq returned no usable text; keeping original")
        return text, []
    changes = [c for c in payload.get("changes", []) if isinstance(c, str)]
    return cleaned, changes
