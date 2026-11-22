# Observability for Simulation Runs

Record structured events for:
- Vehicle arrival and departure.
- Queue entry and exit.
- Charger assignment and release.
- Requested action and feasibility result.
- Energy delivered per time step.
- Station aggregate load.
- Reward components and terminal outcome.

Use a run identifier and simulation timestamp on every event. Avoid logging personal identifiers. Structured logs make it possible to diagnose policy behavior and reconcile aggregate metrics.
