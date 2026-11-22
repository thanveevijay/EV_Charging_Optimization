# Threat Model and Data Handling

This repository is a research prototype. If connected to operational systems, consider:

## Data risks
- Personally identifying session metadata in raw exports.
- Accidental publication of credentials or local configuration.
- Dataset licensing and redistribution restrictions.

## System risks
- Malformed telemetry causing invalid state.
- Stale charger availability leading to conflicting assignments.
- Model output proposing an unavailable or unsafe action.
- Dependency compromise in the Python environment.

## Controls
- Keep raw/private data out of version control.
- Use environment variables or a secret manager for credentials.
- Validate every observation before it enters the model.
- Enforce power and availability constraints in deterministic simulator/controller code.
- Pin and review dependencies before deployment.
