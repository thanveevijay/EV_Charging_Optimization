# Methodology

## 1. Data preparation
Parse timestamps consistently, coerce energy values to numeric, remove invalid intervals, and retain a record of excluded rows.

## 2. Demand features
Aggregate arrivals and energy by hour. Use calendar features and lagged/rolling statistics. Every rolling feature must be shifted by one period before calculation to avoid target leakage.

## 3. Graph representation
Model charging entities and their relationships explicitly. Keep node-feature definitions, edge meaning, and graph construction rules versioned with each experiment.

## 4. Scheduling
The DQN agent observes an environment state and selects an available scheduling action. The simulator applies charger availability and session departure constraints, then returns a reward.

## 5. Baseline
Run FCFS on the same sessions, charger capacity, and time horizon as the learned policy.

## 6. Evaluation
Report averages and distributions, not only a single best run. Repeat stochastic experiments with fixed and varied random seeds.
