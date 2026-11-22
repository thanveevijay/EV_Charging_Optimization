"""Small statistical helpers for summarizing repeated experiment runs."""
from __future__ import annotations

import math
from collections.abc import Iterable


def mean(values: Iterable[float]) -> float:
    data = [float(value) for value in values]
    if not data:
        raise ValueError("At least one value is required.")
    if any(not math.isfinite(value) for value in data):
        raise ValueError("Values must be finite.")
    return sum(data) / len(data)


def sample_std(values: Iterable[float]) -> float:
    data = [float(value) for value in values]
    if len(data) < 2:
        raise ValueError("At least two values are required for sample standard deviation.")
    if any(not math.isfinite(value) for value in data):
        raise ValueError("Values must be finite.")
    center = sum(data) / len(data)
    variance = sum((value - center) ** 2 for value in data) / (len(data) - 1)
    return math.sqrt(variance)


def mean_and_std(values: Iterable[float]) -> dict[str, float]:
    data = [float(value) for value in values]
    return {"mean": mean(data), "sample_std": sample_std(data), "runs": float(len(data))}
