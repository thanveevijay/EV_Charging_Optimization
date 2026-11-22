# Project Overview

## Goal
Optimize the allocation of limited EV charging resources while balancing energy fulfillment, waiting time, charger utilization, and station peak demand.

## Current approach
The repository combines historical ACN session data, preprocessing, demand forecasting, graph construction, a graph encoder, and a Deep Q-Network (DQN) scheduler. FCFS is retained as a transparent baseline.

## Main workflow
1. Validate and clean charging-session records.
2. Aggregate sessions and engineer time-based demand features.
3. Represent EVs and charging resources as a graph.
4. Encode graph state and select scheduling actions.
5. Simulate charging under charger and time constraints.
6. Compare learned scheduling with FCFS using consistent metrics.

## Important interpretation
Historical energy delivered is used as a proxy for demand where a requested-energy field is unavailable. It should not be described as a directly observed unmet-demand label.
