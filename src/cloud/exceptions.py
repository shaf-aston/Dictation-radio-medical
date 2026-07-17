"""Exception hierarchy for the cloud subsystem.

All cloud failures derive from :class:`CloudError` so callers can catch the
whole subsystem with one clause and degrade gracefully to offline operation.
"""

from __future__ import annotations

# PHI de-identification is a shared core-domain concern (used by dictation and
# training, not just cloud), so its failure type lives in src.medical.deid and is
# re-exported here for back-compat with callers that catch it by this name.
from src.medical.deid import PrivacyError  # noqa: F401

# Imaging failures are their own subsystem; re-export the canonical type defined
# in src.imaging.exceptions rather than defining a second, incompatible class of
# the same name here (the old cloud-local ImagingError was never raised/caught).
from src.imaging.exceptions import ImagingError  # noqa: F401


class CloudError(Exception):
    """Base class for all cloud-integration failures."""


class AuthError(CloudError):
    """Missing or invalid Lightning AI credentials."""


class QuotaError(CloudError):
    """Lightning AI compute quota or rate limit exceeded."""


class JobError(CloudError):
    """A training job failed on Lightning AI."""


class GroqError(CloudError):
    """A Groq LLM cleanup request failed or returned unusable output."""
