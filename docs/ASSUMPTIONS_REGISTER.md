# Assumptions Register

| ID | Assumption | Why it matters | Validation action |
|---|---|---|---|
| A1 | Historical delivered energy approximates session demand when requested energy is absent | Can bias fulfillment estimates | Label it as a proxy and seek requested-energy data |
| A2 | A charging space maps to one charger | Determines resource capacity | Confirm station metadata |
| A3 | Charger power is constant in a simulation run | Simplifies energy accounting | Compare with charger specifications |
| A4 | Time is discretized into fixed steps | Affects queueing and energy delivery | Run a time-step sensitivity study |
| A5 | Historical sessions are representative | Affects generalization | Evaluate on later time periods and other stations |

Update this register whenever an assumption changes. Results should identify which assumptions were active.
