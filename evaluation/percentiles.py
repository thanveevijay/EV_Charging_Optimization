"""Percentile summaries for skewed waiting-time distributions."""
from __future__ import annotations

import math
from collections.abc import Iterable


def percentile(values: Iterable[float], quantile: float) -> float:
    data = sorted(float(value) for value in values)
    q = float(quantile)
    if not data:
        raise ValueError("At least one value is required.")
    if any(not math.isfinite(value) for value in data):
        raise ValueError("Values must be finite.")
    if not 0 <= q <= 1:
        raise ValueError("quantile must be between 0 and 1.")
    position = (len(data) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return data[lower]
    fraction = position - lower
    return data[lower] * (1 - fraction) + data[upper] * fraction
