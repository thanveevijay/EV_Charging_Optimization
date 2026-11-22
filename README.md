# AI-Based Dynamic EV Charging Optimization

A research prototype for scheduling electric-vehicle charging sessions under limited charger and station-power resources.

## Core approach
- Graph Neural Network (GNN) representations of charging-station state.
- Deep Q-Network (DQN) scheduling experiments.
- First-Come, First-Served (FCFS) baseline.
- Historical demand preprocessing and forecasting.
- Discrete-time charging simulation and comparative evaluation.

## Repository structure
- `data_preprocessing.py`, `predict_demand.py`: data preparation and demand modeling.
- `graph_construction.py`, `gnn_model.py`: graph representation and encoder.
- `dqn_agent.py`, `train_agent.py`: reinforcement-learning components.
- `charging_environment.py`, `charging_simulator.py`: simulation environments.
- `fcfs_baseline.py`: baseline scheduling.
- `evaluation/`: reusable metrics and statistical summaries.
- `scheduling/`: deterministic priority and feasibility helpers.
- `scripts/`: dataset inspection and validation utilities.
- `tests/`: lightweight unit tests.
- `docs/`: methodology, assumptions, evaluation, safety, and reproducibility notes.

## Setup
Use Python 3.10 or 3.11 in a virtual environment. Install the dependencies in `requirements.txt`. PyTorch Geometric may require a platform-specific installation compatible with the installed PyTorch version.

## Data
The project expects ACN-style charging session data. Obtain the dataset separately and comply with its terms. Do not commit raw/private session exports. See [Data Dictionary](docs/DATA_DICTIONARY.md) and [Privacy Guidelines](docs/PRIVACY_GUIDELINES.md).

## Lightweight checks
```bash
python -m unittest discover -s tests -v
python scripts/validate_processed_data.py outputs/results/processed_acn_data.csv
```

## Research integrity
Historical delivered energy may be used as a proxy for demand when requested-energy labels are unavailable. This is not equivalent to observed requested energy. Report assumptions, chronological splits, repeated-run variation, and limitations with every result.

This repository is a simulation and research prototype, not a production charging controller. Real deployments require independent electrical safety enforcement, hardware-in-the-loop testing, operator override, and regulatory review.
