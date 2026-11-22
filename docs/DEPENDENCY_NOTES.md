# Dependency Notes

The main project uses scientific Python packages and PyTorch-based model components. The exact versions should be pinned after a clean environment has been tested.

## Suggested environment checks
```bash
python --version
python -m pip check
python -m pip freeze
```

PyTorch Geometric installation can depend on the installed PyTorch version and platform. Follow the official compatibility instructions for the target environment rather than assuming a single wheel works everywhere.

Keep optional visualization and training dependencies separate from lightweight metric tests where practical.
