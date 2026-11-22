"""Print a compact quality report for a charging-session CSV."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def inspect(path: Path) -> dict[str, object]:
    frame = pd.read_csv(path)
    report: dict[str, object] = {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_by_column": frame.isna().sum().to_dict(),
    }
    if "connectionTime" in frame:
        timestamps = pd.to_datetime(frame["connectionTime"], utc=True, errors="coerce")
        report["invalid_connection_timestamps"] = int(timestamps.isna().sum())
        if timestamps.notna().any():
            report["connection_time_min"] = str(timestamps.min())
            report["connection_time_max"] = str(timestamps.max())
    if "kWhDelivered" in frame:
        energy = pd.to_numeric(frame["kWhDelivered"], errors="coerce")
        report["invalid_energy_values"] = int(energy.isna().sum())
        report["negative_energy_values"] = int((energy.dropna() < 0).sum())
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Session CSV to inspect")
    args = parser.parse_args()
    if not args.csv.exists():
        parser.error(f"File not found: {args.csv}")
    for key, value in inspect(args.csv).items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
