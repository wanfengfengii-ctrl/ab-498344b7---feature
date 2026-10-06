"""Stable, machine-readable API error type."""
from __future__ import annotations

from typing import Optional


class ApiError(Exception):
    """Client-facing error carrying a stable code and a locatable field path.

    ``code``   -- stable machine-readable identifier (see README for the full list)
    ``field``  -- path to the offending element, e.g. ``inputs[2].value`` or
                  ``covariance[0][1]``; ``None`` when not applicable
    ``status`` -- HTTP status code to return
    """

    def __init__(self, code: str, message: str,
                 field: Optional[str] = None, status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.field = field
        self.status = status

    def payload(self) -> dict:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "field": self.field,
            }
        }
