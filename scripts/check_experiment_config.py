"""Validate essential fields in a YAML experiment configuration."""
from __future__ import annotations

import argparse
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


def validate_config(data: dict) -> list[str]:
    errors = []
    if not isinstance(data, dict):
        return ["Configuration root must be a mapping"]
    for section in ("experiment", "simulation", "policy"):
        if section not in data or not isinstance(data[section], dict):
            errors.append(f"Missing mapping section: {section}")
    simulation = data.get("simulation", {})
    if isinstance(simulation, dict):
        for key in ("charger_power_kw", "time_step_minutes"):
            value = simulation.get(key)
            if not isinstance(value, (int, float)) or value <= 0:
                errors.append(f"simulation.{key} must be positive")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    if yaml is None:
        parser.error("Install PyYAML to validate YAML files")
    data = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    errors = validate_config(data)
    if errors:
        print("\n".join(errors))
        return 1
    print("Experiment configuration is structurally valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
