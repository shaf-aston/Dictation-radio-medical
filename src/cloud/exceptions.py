"""Exception hierarchy for the cloud subsystem.

All cloud failures derive from :class:`CloudError` so callers can catch the
whole subsystem with one clause and degrade gracefully to offline operation.
"""

from __future__ import annotations


class CloudError(Exception):
    """Base class for all cloud-integration failures."""


class AuthError(CloudError):
    """Missing or invalid Lightning AI credentials."""


class QuotaError(CloudError):
    """Lightning AI compute quota or rate limit exceeded."""


class PrivacyError(CloudError):
    """De-identification failed validation — data must NOT leave the device."""


class JobError(CloudError):
    """A training job failed on Lightning AI."""
