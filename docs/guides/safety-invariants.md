# Safety Invariants

- No connector serves overlapping sessions.
- Site power limits are never exceeded.
- Energy allocations remain non-negative.
- Cancelled sessions release reserved capacity.
- Invalid telemetry cannot silently increase a power allocation.

Run invariant checks after every scheduling decision in simulation tests.