# Arrival Process

Represent vehicle arrival as an explicit event with a session identifier and timestamp. Validate the requested energy and compatible connector before adding the session to the scheduling queue.

If a vehicle arrives late or its request changes, recompute feasibility rather than preserving an outdated assignment. Keep the original request and subsequent changes available for experiment auditing.