"""Demand profile summaries for station-hour observations."""
from __future__ import annotations

from collections.abc import Iterable


def demand_summary(arrivals: Iterable[int], energy_kwh: Iterable[float]) -> dict[str, float]:
    arrival_values = [int(x) for x in arrivals]
    energy_values = [float(x) for x in energy_kwh]
    if not arrival_values or not energy_values:
        raise ValueError("Demand series cannot be empty")
    if len(arrival_values) != len(energy_values):
        raise ValueError("Series lengths must match")
    if any(x < 0 for x in arrival_values) or any(x < 0 for x in energy_values):
        raise ValueError("Demand values cannot be negative")
    return {
        "periods": float(len(arrival_values)),
        "total_arrivals": float(sum(arrival_values)),
        "mean_arrivals": sum(arrival_values) / len(arrival_values),
        "total_energy_kwh": sum(energy_values),
        "mean_energy_kwh": sum(energy_values) / len(energy_values),
    }
