# Reward Design Notes

A scheduling reward can combine service quality and grid impact. A general form is:

`R_t = w_e E_t - w_w W_t - w_p P_t - w_u U_t`

where:
- `E_t`: energy successfully delivered during the step.
- `W_t`: waiting-time penalty.
- `P_t`: peak-load or load-imbalance penalty.
- `U_t`: penalty for infeasible or invalid actions.
- `w_*`: non-negative weights selected using validation experiments.

## Design checks
- Keep reward terms on comparable scales.
- Penalize invalid actions consistently.
- Avoid rewarding energy delivery in a way that encourages violating departure deadlines.
- Record each reward component separately for debugging.
- Run sensitivity analysis over the weights; do not tune on the final test period.
