"""Build action masks for available charging points."""
from __future__ import annotations


def available_action_mask(availability: list[bool]) -> list[int]:
    if not isinstance(availability, list):
        raise TypeError("availability must be a list")
    return [1 if bool(value) else 0 for value in availability]


def validate_action(action: int, mask: list[int]) -> bool:
    if not isinstance(action, int) or action < 0 or action >= len(mask):
        return False
    return mask[action] == 1
