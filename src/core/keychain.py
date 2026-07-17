"""OS-keychain secret storage (via ``keyring``).

Every integration that holds an API key (Lightning AI, Groq, ...) repeated the
same store/get/clear-with-keyring skeleton under its own service name. This
module is that skeleton, written once: callers supply only the key name their
secret is stored under.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_SERVICE = "radio-dictate"


def store_secret(key: str, value: str) -> None:
    """Persist *value* under *key* in the OS keychain."""
    import keyring
    keyring.set_password(_SERVICE, key, value)


def get_secret(key: str) -> Optional[str]:
    """Return the secret stored under *key*, or None if unset/unreadable."""
    try:
        import keyring
        return keyring.get_password(_SERVICE, key)
    except Exception as exc:
        logger.warning("Could not read %s from keychain: %s", key, exc)
        return None


def clear_secret(key: str) -> None:
    """Remove the secret stored under *key*, if any."""
    try:
        import keyring
        keyring.delete_password(_SERVICE, key)
    except Exception as exc:
        logger.debug("Could not clear %s from keychain: %s", key, exc)
