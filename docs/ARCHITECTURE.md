# System Architecture

The project is organized as a research pipeline:

```text
ACN session data
      |
      v
Data validation and preprocessing
      |
      +------> Hourly demand forecasting
      |
      v
Charging-station graph construction
      |
      v
Graph representation / GNN encoder
      |
      v
DQN scheduling policy <----> Charging simulator
      |                           |
      +---------- metrics <-------+
                    |
                    v
             FCFS comparison
```

## Boundaries
- Data preparation owns schema validation and feature generation.
- The simulator owns resource availability and physical constraints.
- The policy proposes actions but must not bypass simulator constraints.
- Evaluation computes metrics from completed runs and does not alter policy behavior.

This separation makes it easier to test components independently and compare alternative policies fairly.
