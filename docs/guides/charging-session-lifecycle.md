# Charging Session Lifecycle

A session can move through `requested`, `queued`, `assigned`, `charging`, `completed`, or `cancelled`.

## Invariants
- A vehicle cannot occupy two connectors simultaneously.
- A connector cannot serve overlapping sessions.
- Completed sessions have an end time no earlier than their start time.
- Cancelled sessions release reserved capacity.
- State transitions are logged for later audit.