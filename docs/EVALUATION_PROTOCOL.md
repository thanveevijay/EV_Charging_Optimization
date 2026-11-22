# Evaluation Protocol

Compare the learned policy with FCFS using identical input sessions and resource limits.

## Primary metrics
- Mean and median waiting time (minutes).
- Energy fulfillment ratio: delivered energy divided by requested energy, only when requested energy is available.
- Session completion rate.
- Charger utilization.
- Peak aggregate charging load (kW).

## Reporting rules
1. State the number of sessions and chargers.
2. State charger power and simulation time-step.
3. Report the evaluation date range and preprocessing version.
4. Use multiple random seeds for stochastic policies.
5. Include mean, standard deviation, and sample count.
6. Avoid claiming statistical significance without an appropriate test.

## Fairness
A scheduling policy should not be evaluated on a different demand sample or a more favorable charger configuration than its baseline.
