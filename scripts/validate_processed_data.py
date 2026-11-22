"""Validate the processed ACN-style CSV before model training."""
from pathlib import Path
import argparse
import pandas as pd


REQUIRED = {"connectionTime", "kWhDelivered"}


def validate(path: Path) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not path.exists():
        return False, [f"File does not exist: {path}"]

    frame = pd.read_csv(path)
    missing = REQUIRED.difference(frame.columns)
    if missing:
        return False, [f"Missing required columns: {sorted(missing)}"]

    timestamps = pd.to_datetime(frame["connectionTime"], utc=True, errors="coerce")
    energy = pd.to_numeric(frame["kWhDelivered"], errors="coerce")

    if timestamps.isna().any():
        errors.append(f"{int(timestamps.isna().sum())} invalid timestamps")
    if energy.isna().any():
        errors.append(f"{int(energy.isna().sum())} non-numeric energy values")
    if (energy.dropna() < 0).any():
        errors.append("Negative energy values found")
    if frame.empty:
        errors.append("Dataset contains no rows")

    return not errors, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv",
        nargs="?",
        default="outputs/results/processed_acn_data.csv",
        help="Path to processed session CSV",
    )
    args = parser.parse_args()
    valid, errors = validate(Path(args.csv))

    if valid:
        print("Dataset validation passed.")
        return 0

    print("Dataset validation failed:")
    for error in errors:
        print(f"- {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
