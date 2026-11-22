# Load Forecasting

## Inputs
Use historical station demand, time of day, day of week, local events when available, and weather only when its provenance is known.

## Evaluation
Use chronological train/validation/test splits. Compare against persistence and seasonal-naive baselines. Report MAE and RMSE alongside peak-period error.

Avoid random shuffling across time because it can leak future patterns into training.