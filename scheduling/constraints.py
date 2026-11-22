"""Feasibility checks for proposed charging assignments."""
from __future__ import annotations


def can_assign(
    charger_available: bool,
    vehicle_compatible: bool,
    station_load_kw: float,
    requested_power_kw: float,
    station_limit_kw: float,
) -> bool:
    """Return whether an assignment respects basic availability and power limits."""
    values = (station_load_kw, requested_power_kw, station_limit_kw)
    if any(value < 0 for value in values):
        raise ValueError("Power values cannot be negative.")
    if requested_power_kw == 0 or station_limit_kw == 0:
        return False
    return (
        bool(charger_available)
        and bool(vehicle_compatible)
        and station_load_kw + requested_power_kw <= station_limit_kw
    )
