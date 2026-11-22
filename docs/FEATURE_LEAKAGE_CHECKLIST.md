# Feature Leakage Checklist

Before training a demand model, verify:

- [ ] The chronological split is created before fitting transformations.
- [ ] Lag features use only earlier timestamps.
- [ ] Rolling statistics are shifted before aggregation.
- [ ] No future session departure or delivered-energy value is used to predict an earlier time.
- [ ] Imputation parameters are fitted on training data only.
- [ ] Hyperparameters are selected using validation data, not the test period.
- [ ] The final test period is evaluated once after model selection.

For time-series forecasting, random row splitting can leak temporal information and should not be used without a clear justification.
