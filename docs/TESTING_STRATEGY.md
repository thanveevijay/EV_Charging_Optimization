# Testing Strategy

## Data tests
- Required columns exist.
- Timestamps parse and departure follows arrival.
- Energy values are finite and non-negative.
- Empty or fully invalid datasets fail with actionable messages.

## Simulation tests
- A session cannot charge before arrival.
- A charger cannot serve two sessions simultaneously.
- Delivered energy never exceeds the configured feasible energy for a time step.
- Departed sessions are no longer active.
- No action can allocate a non-existent charger.

## Model tests
- State dimensions match the model input.
- Action selection respects the valid-action mask.
- Training and evaluation modes are separated.
- Checkpoints can be loaded and evaluated.

Use small synthetic fixtures so core tests do not require downloading the full ACN dataset.
