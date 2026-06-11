"""Exception hierarchy for the imaging subsystem.

All imaging failures derive from :class:`ImagingError` so callers can catch the
whole subsystem with one clause and degrade gracefully to offline operation.
"""

from __future__ import annotations


class ImagingError(Exception):
    """Base class for all imaging-subsystem failures."""
