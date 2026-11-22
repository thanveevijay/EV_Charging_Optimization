"""Deterministic priority scores for rule-based EV scheduling."""
from __future__ import annotations

import math


def earliest_departure_score(minutes_to_departure: float) -> float:
    """Higher score means more urgent; zero departure time is most urgent."""
    value = float(minutes_to_departure)
    if not math.isfinite(value) or value < 0:
        raise ValueError("minutes_to_departure must be finite and non-negative.")
    return 1.0 / (1.0 + value)


def energy_need_score(remaining_kwh: float, available_power_kw: float) -> float:
    """Estimate normalized urgency from remaining energy and available power."""
    energy = float(remaining_kwh)
    power = float(available_power_kw)
    if not math.isfinite(energy) or not math.isfinite(power):
        raise ValueError("Inputs must be finite.")
    if energy < 0 or power <= 0:
        raise ValueError("Energy must be non-negative and power must be positive.")
    estimated_hours = energy / power
    return estimated_hours / (1.0 + estimated_hours)


def combined_priority(minutes_to_departure: float,
                      remaining_kwh: float,
                      available_power_kw: float,
                      departure_weight: float = 0.5) -> float:
    """Return a weighted score; larger values indicate higher priority."""
    weight = float(departure_weight)
    if not 0 <= weight <= 1:
        raise ValueError("departure_weight must be between 0 and 1.")
    departure = earliest_departure_score(minutes_to_departure)
    energy = energy_need_score(remaining_kwh, available_power_kw)
    return weight * departure + (1.0 - weight) * energy
