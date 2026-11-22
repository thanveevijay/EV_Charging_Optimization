# Scenario Design

Use controlled scenarios to stress different operating conditions:

- Low utilization: arrivals are sparse relative to charger count.
- Peak arrivals: many EVs arrive within a short interval.
- Long dwell: vehicles remain connected for extended periods.
- Short dwell: departure deadlines are tight.
- Power constrained: station cap is below the sum of charger ratings.
- Heterogeneous demand: energy needs and vehicle acceptance limits vary.

For each scenario, specify arrival generation, demand distribution, charger constraints, random seed, and expected invariants before evaluating policies.
