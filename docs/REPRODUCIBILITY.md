# Reproducibility Checklist

- [ ] Record the Git commit SHA used for the run.
- [ ] Record dataset source, version, and checksum.
- [ ] Record Python and package versions.
- [ ] Set and log Python, NumPy, and framework random seeds.
- [ ] Save train, validation, and test time ranges.
- [ ] Fit preprocessing steps on training data only.
- [ ] Save model configuration and checkpoint.
- [ ] Save evaluation configuration and raw metrics.
- [ ] Run FCFS on the same evaluation sessions.
- [ ] Note hardware and accelerator details where relevant.

A result is reproducible only when both the data preparation and the simulation configuration are recorded.
