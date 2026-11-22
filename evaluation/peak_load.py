"""Peak-load and load-factor helpers for station-level evaluation."""
from __future__ import annotations

import math
from collections.abc import Iterable


def peak_load_kw(loads_kw: Iterable[float]) -> float:
    values = [float(value) for value in loads_kw]
    if not values:
        raise ValueError("At least one load value is required.")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Loads must be finite and non-negative.")
    return max(values)


def average_load_kw(loads_kw: Iterable[float]) -> float:
    values = [float(value) for value in loads_kw]
    if not values:
        raise ValueError("At least one load value is required.")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Loads must be finite and non-negative.")
    return sum(values) / len(values)


def load_factor(loads_kw: Iterable[float]) -> float:
    values = [float(value) for value in loads_kw]
    peak = peak_load_kw(values)
    if peak == 0:
        return 0.0
    return average_load_kw(values) / peak
