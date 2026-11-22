"""Normal-approximation confidence interval for repeated-run means."""
from __future__ import annotations

import math
from collections.abc import Iterable


def mean_confidence_interval(values: Iterable[float], z_value: float = 1.96) -> tuple[float, float]:
    data = [float(value) for value in values]
    if len(data) < 2:
        raise ValueError("At least two runs are required")
    if any(not math.isfinite(value) for value in data):
        raise ValueError("Values must be finite")
    z = float(z_value)
    if z <= 0:
        raise ValueError("z_value must be positive")
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
    margin = z * math.sqrt(variance / len(data))
    return mean - margin, mean + margin
