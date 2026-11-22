"""Greedy allocation helper for a bounded station power budget."""
from __future__ import annotations


def remaining_station_capacity(limit_kw: float, active_load_kw: float) -> float:
    limit = float(limit_kw)
    active = float(active_load_kw)
    if limit < 0 or active < 0:
        raise ValueError("Power values must be non-negative")
    return max(0.0, limit - active)


def clamp_requested_power(request_kw: float, charger_limit_kw: float,
                          vehicle_limit_kw: float, station_remaining_kw: float) -> float:
    values = [float(request_kw), float(charger_limit_kw),
              float(vehicle_limit_kw), float(station_remaining_kw)]
    if any(value < 0 for value in values):
        raise ValueError("Power values must be non-negative")
    return min(values)
