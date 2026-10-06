"""Strict parsing of rationals.

Every rational in the API must be either

* a JSON integer (e.g. ``3`` or ``"-7"``), or
* a string ``"p/q"`` in **lowest terms** with a **positive** denominator.

Anything else (floats, decimals, scientific notation, booleans, unreduced
fractions, non-positive denominators, ...) is rejected so that no floating
point approximation can ever leak into a release decision.
"""
from __future__ import annotations

import re
from fractions import Fraction
from math import gcd
from typing import Any

_INT_RE = re.compile(r"[+-]?\d+")
_FRAC_RE = re.compile(r"([+-]?\d+)/([+-]?\d+)")

# Generous bound to keep pathological inputs from exhausting CPU/memory.
MAX_LITERAL_LENGTH = 1024


class RationalFormatError(ValueError):
    """Raised when a JSON node is not an allowed rational literal."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def parse_rational(node: Any) -> Fraction:
    """Parse a JSON node into an exact :class:`Fraction`.

    Raises :class:`RationalFormatError` with a human-readable reason when the
    node is not an integer or a reduced ``p/q`` string with q > 0.
    """
    # bool is a subclass of int -- reject it explicitly.
    if isinstance(node, bool):
        raise RationalFormatError("booleans are not valid rationals")
    if isinstance(node, int):
        return Fraction(node)
    if isinstance(node, float):
        raise RationalFormatError(
            "decimal/scientific notation is not allowed; "
            "use an integer or a 'p/q' string"
        )
    if isinstance(node, str):
        if not node:
            raise RationalFormatError("empty string is not a rational")
        if len(node) > MAX_LITERAL_LENGTH:
            raise RationalFormatError("rational literal is too long")
        if _INT_RE.fullmatch(node):
            return Fraction(int(node))
        match = _FRAC_RE.fullmatch(node)
        if match is not None:
            numerator = int(match.group(1))
            denominator = int(match.group(2))
            if denominator <= 0:
                raise RationalFormatError("denominator must be positive")
            if gcd(abs(numerator), denominator) != 1:
                raise RationalFormatError("fraction is not in lowest terms")
            return Fraction(numerator, denominator)
        raise RationalFormatError(
            "expected an integer or a 'p/q' string with a positive denominator"
        )
    raise RationalFormatError(
        "unsupported JSON type '%s' for a rational" % type(node).__name__
    )


def format_rational(value: Fraction) -> str:
    """Render a Fraction in its canonical reduced form (``3`` or ``3/4``)."""
    return str(value)
