# Data Dictionary

The preprocessing and simulation scripts use ACN-style charging-session records. Confirm exact column names against the source dataset before running a pipeline.

| Field | Meaning | Validation |
|---|---|---|
| sessionID | Session identifier | Treat as an identifier, not a numeric feature |
| spaceID | Charging-space identifier | Non-empty |
| connectionTime | Session arrival timestamp | Parse to timezone-aware datetime |
| disconnectTime | Session departure timestamp | Must be later than arrival |
| kWhDelivered | Energy delivered in the session | Numeric and non-negative |

## Derived fields
- `hour`: arrival timestamp rounded down to the hour.
- `ev_arrivals`: number of sessions arriving in an hour.
- `energy_demand_kwh`: sum of delivered energy used as a demand proxy.
- Lag and rolling features: historical values shifted so future observations are not used.

## Data caveats
Do not assume delivered energy equals requested energy. Document timezone handling, missing-value policy, duplicate handling, and any filtering thresholds in experiment reports.
