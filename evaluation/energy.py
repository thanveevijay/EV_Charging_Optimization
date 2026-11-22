"""Energy accounting helpers with explicit unit conversions."""
from __future__ import annotations


def kw_minutes_to_kwh(power_kw: float, minutes: float) -> float:
    power = float(power_kw)
    duration = float(minutes)
    if power < 0 or duration < 0:
        raise ValueError("Power and duration must be non-negative")
    return power * duration / 60.0


def fulfillment_percentage(delivered_kwh: float, requested_kwh: float) -> float:
    delivered = float(delivered_kwh)
    requested = float(requested_kwh)
    if delivered < 0 or requested <= 0:
        raise ValueError("Delivered energy must be non-negative and requested energy positive")
    return 100.0 * delivered / requested
