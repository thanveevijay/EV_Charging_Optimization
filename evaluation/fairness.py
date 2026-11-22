"""Group-wise service metrics for charging-session evaluations."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable


def grouped_mean_wait(groups: Iterable[str], waits: Iterable[float]) -> dict[str, float]:
    group_values = list(groups)
    wait_values = [float(value) for value in waits]
    if len(group_values) != len(wait_values):
        raise ValueError("groups and waits must have equal lengths")
    if not group_values:
        raise ValueError("At least one observation is required")
    if any(value < 0 for value in wait_values):
        raise ValueError("Waiting times cannot be negative")
    buckets: dict[str, list[float]] = defaultdict(list)
    for group, wait in zip(group_values, wait_values):
        buckets[str(group)].append(wait)
    return {key: sum(values) / len(values) for key, values in sorted(buckets.items())}
