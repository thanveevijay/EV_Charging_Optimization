"""Waiting-time distribution helpers."""
from __future__ import annotations

from collections.abc import Iterable


def waiting_rate(waiting_minutes: Iterable[float], threshold_minutes: float) -> float:
    values = [float(x) for x in waiting_minutes]
    threshold = float(threshold_minutes)
    if not values:
        raise ValueError("At least one wait value is required")
    if threshold < 0 or any(x < 0 for x in values):
        raise ValueError("Waiting times and threshold must be non-negative")
    return sum(x > threshold for x in values) / len(values)


def zero_wait_share(waiting_minutes: Iterable[float]) -> float:
    values = [float(x) for x in waiting_minutes]
    if not values:
        raise ValueError("At least one wait value is required")
    if any(x < 0 for x in values):
        raise ValueError("Waiting times cannot be negative")
    return sum(x == 0 for x in values) / len(values)
