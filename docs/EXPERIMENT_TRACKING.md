# Experiment Tracking

Record one row per experiment.

| Field | Description |
|---|---|
| experiment_id | Unique run identifier |
| code_revision | Git commit SHA |
| dataset_version | Dataset source and checksum |
| split | Train/validation/test time ranges |
| seed | Random seed |
| policy | FCFS, DQN, or other policy |
| charger_count | Number of charging points |
| charger_power_kw | Per-charger power |
| time_step_minutes | Simulation resolution |
| reward_config | Reward weights and constraints |
| metrics | Waiting, completion, energy, utilization, peak load |

Never overwrite a prior result. Store configuration and metrics alongside each run so comparisons can be reproduced.
