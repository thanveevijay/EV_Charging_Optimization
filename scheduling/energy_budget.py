"""Energy delivery bounds for a fixed simulation time step."""
from __future__ import annotations


def maximum_step_energy_kwh(
    charger_power_kw: float,
    step_minutes: float,
    vehicle_acceptance_kw: float | None = None,
) -> float:
    charger_power = float(charger_power_kw)
    duration = float(step_minutes)
    if charger_power <= 0 or duration <= 0:
        raise ValueError("Charger power and time step must be positive.")
    effective_power = charger_power
    if vehicle_acceptance_kw is not None:
        acceptance = float(vehicle_acceptance_kw)
        if acceptance <= 0:
            raise ValueError("Vehicle acceptance power must be positive.")
        effective_power = min(charger_power, acceptance)
    return effective_power * duration / 60.0


def energy_shortfall_kwh(target_kwh: float, delivered_kwh: float) -> float:
    target = float(target_kwh)
    delivered = float(delivered_kwh)
    if target < 0 or delivered < 0:
        raise ValueError("Energy values cannot be negative.")
    return max(0.0, target - delivered)
