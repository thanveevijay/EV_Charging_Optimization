# Deployment Readiness Gate

This repository currently demonstrates research and simulation workflows. Before any live use, require:

- Verified station telemetry and charger state.
- A validated mapping from model actions to charger commands.
- Strict power and electrical safety constraints enforced outside the model.
- A deterministic fallback scheduler.
- Timeout, retry, and communication-loss handling.
- Human operator override and audit logs.
- Load testing and hardware-in-the-loop validation.
- Security, privacy, and regulatory review.

The learned policy must never be the sole enforcement point for electrical safety limits.
