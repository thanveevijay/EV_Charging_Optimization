"""Evaluation metrics for EV charging scheduling experiments.

All functions accept plain numeric sequences and avoid framework dependencies.
"""
from __future__ import annotations

from typing import Iterable
import math


def _finite_values(values: Iterable[float]) -> list[float]:
    cleaned = [float(value) for value in values]
    if any(not math.isfinite(value) for value in cleaned):
        raise ValueError("Metric inputs must contain only finite values.")
    return cleaned


def waiting_time_summary(waiting_minutes: Iterable[float]) -> dict[str, float]:
    values = _finite_values(waiting_minutes)
    if not values:
        raise ValueError("At least one waiting-time value is required.")
    if any(value < 0 for value in values):
        raise ValueError("Waiting times cannot be negative.")
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (
        ordered[middle - 1] + ordered[middle]
    ) / 2
    return {
        "count": float(len(values)),
        "mean_minutes": sum(values) / len(values),
        "median_minutes": median,
        "max_minutes": max(values),
    }


def completion_rate(completed: int, total: int) -> float:
    if total <= 0:
        raise ValueError("total must be greater than zero.")
    if completed < 0 or completed > total:
        raise ValueError("completed must be between zero and total.")
    return completed / total


def energy_fulfillment_ratio(delivered_kwh: float, requested_kwh: float) -> float:
    delivered = float(delivered_kwh)
    requested = float(requested_kwh)
    if not math.isfinite(delivered) or not math.isfinite(requested):
        raise ValueError("Energy values must be finite.")
    if delivered < 0 or requested <= 0:
        raise ValueError("Delivered energy must be non-negative and requested energy positive.")
    return delivered / requested


def charger_utilization(busy_charger_minutes: float,
                        available_charger_minutes: float) -> float:
    busy = float(busy_charger_minutes)
    available = float(available_charger_minutes)
    if not math.isfinite(busy) or not math.isfinite(available):
        raise ValueError("Utilization inputs must be finite.")
    if available <= 0 or busy < 0 or busy > available:
        raise ValueError("Require 0 <= busy <= available and available > 0.")
    return busy / available
