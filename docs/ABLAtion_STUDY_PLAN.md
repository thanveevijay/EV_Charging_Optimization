# Ablation Study Plan

Ablations help determine which parts of the scheduling approach contribute to observed outcomes.

Suggested controlled comparisons:

1. FCFS versus DQN without graph features.
2. DQN with graph encoder versus DQN with flat state features.
3. Full reward versus reward without peak-load penalty.
4. Full state versus state without demand-forecast features.
5. Different graph neighborhood definitions.
6. Different simulation time steps.

For each comparison, keep the dataset split, charger configuration, training budget, and evaluation seeds fixed. Report variability and avoid changing multiple factors in one ablation.
