"""Small deterministic FIFO queue for simulation components."""
from __future__ import annotations

from collections import deque
from typing import Generic, TypeVar

T = TypeVar("T")


class ArrivalQueue(Generic[T]):
    def __init__(self) -> None:
        self._items: deque[T] = deque()

    def add(self, item: T) -> None:
        self._items.append(item)

    def pop_next(self) -> T:
        if not self._items:
            raise IndexError("Cannot pop from an empty arrival queue")
        return self._items.popleft()

    def __len__(self) -> int:
        return len(self._items)

    def is_empty(self) -> bool:
        return not self._items
