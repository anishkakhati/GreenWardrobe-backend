"""Shared validation helpers for normalized material compositions."""

from decimal import Decimal, InvalidOperation
from typing import Iterable


COMPOSITION_TOTAL_TOLERANCE_PERCENT = Decimal("0.01")


def composition_total_is_valid(percentages: Iterable[object]) -> bool:
    """Return whether percentages sum to 100 within an absolute 0.01% tolerance.

    Decimal conversion via string preserves the represented decimal input and
    avoids widening the tolerance through floating-point relative tolerance.
    """
    try:
        total = sum((Decimal(str(value)) for value in percentages), Decimal("0"))
    except (InvalidOperation, TypeError, ValueError):
        return False
    return total.is_finite() and abs(total - Decimal("100")) <= COMPOSITION_TOTAL_TOLERANCE_PERCENT
