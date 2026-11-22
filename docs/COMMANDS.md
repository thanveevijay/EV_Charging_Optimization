# Common Commands

Run commands from the repository root.

## Validate processed records
```bash
python scripts/validate_processed_data.py outputs/results/processed_acn_data.csv
```

## Inspect source data
```bash
python scripts/inspect_dataset.py path/to/session-data.csv
```

## Run lightweight tests
```bash
python -m unittest discover -s tests -v
```

## Existing project scripts
Use the script-specific configuration and expected input paths before running preprocessing, training, or simulation. Training may require the ACN dataset and optional PyTorch Geometric dependencies.
