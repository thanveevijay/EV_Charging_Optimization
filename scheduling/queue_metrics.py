"""Queue-level metrics for discrete-event scheduling experiments."""
from __future__ import annotations

from collections.abc import Iterable


def average_queue_length(queue_lengths: Iterable[int]) -> float:
    values = [int(value) for value in queue_lengths]
    if not values:
        raise ValueError("At least one queue observation is required.")
    if any(value < 0 for value in values):
        raise ValueError("Queue lengths cannot be negative.")
    return sum(values) / len(values)


def maximum_queue_length(queue_lengths: Iterable[int]) -> int:
    values = [int(value) for value in queue_lengths]
    if not values:
        raise ValueError("At least one queue observation is required.")
    if any(value < 0 for value in values):
        raise ValueError("Queue lengths cannot be negative.")
    return max(values)
