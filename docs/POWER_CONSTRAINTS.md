# Power Constraint Notes

For a charger with rated power (P) kW and a time step of (t) minutes, the ideal upper bound on energy during one step is:

`E_step = P * (t / 60)` kWh.

For example, a 7.2 kW charger over a five-minute interval has an ideal upper bound of 0.6 kWh before efficiency losses or vehicle-side limits.

A station-level cap must be enforced across all simultaneously active chargers. The simulator should also account for connector compatibility, vehicle acceptance power, and any time-varying power-sharing rules when those data are available.
