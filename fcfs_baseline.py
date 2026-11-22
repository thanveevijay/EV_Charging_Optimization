
import numpy as np
import pandas as pd
import torch

from pathlib import Path

from charging_environment import EVChargingEnvironment
from gnn_model import POWER_LEVELS


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = Path("outputs/results/processed_acn_data.csv")

OUTPUT_PATH = Path("outputs/results/fcfs_baseline_results.csv")

SITE_POWER_LIMIT = 30.0
CHARGER_POWER_KW = 7.2
TIME_STEP_MINUTES = 5
NUM_CHARGERS = 54

# Use the final 30% of chronological dates for testing.
TEST_RATIO = 0.30


# ============================================================
# 2. FCFS POLICY
# ============================================================

def select_fcfs_action(env, state):
    """
    Select the earliest-arriving EV that has at least
    one feasible charging-power action.

    For that EV, select the highest feasible power.
    """

    if state is None:
        return None

    _, _, action_mask = state

    # Convert PyTorch mask to NumPy for selection.
    mask = action_mask.cpu().numpy()

    feasible_rows = np.where(mask.any(axis=1))[0]

    if len(feasible_rows) == 0:
        return None

    # Map graph rows to original session indices.
    candidates = []

    for row in feasible_rows:

        session_index = env.unscheduled_indices[row]

        arrival_time = env.session_arrival[session_index]
        departure_time = env.session_departure[session_index]

        candidates.append(
            (arrival_time, departure_time, row)
        )

    # FCFS: earliest arrival first.
    # Departure time is only a deterministic tie-breaker.
    candidates.sort(key=lambda item: (item[0], item[1]))

    selected_row = candidates[0][2]

    # POWER_LEVELS is ascending:
    # 2.4, 4.8, 7.2 kW.
    # Select the highest feasible power.
    feasible_power_indices = np.where(
        mask[selected_row]
    )[0]

    selected_power_index = int(
        feasible_power_indices[-1]
    )

    return selected_row, selected_power_index


# ============================================================
# 3. RUN ONE FCFS EPISODE
# ============================================================

def run_fcfs_episode(day_data, episode_date):

    env = EVChargingEnvironment(
        day_data,
        site_power_limit=SITE_POWER_LIMIT,
        time_step=TIME_STEP_MINUTES,
        charger_power=CHARGER_POWER_KW,
        num_chargers=NUM_CHARGERS
    )

    state = env.reset()

    decisions = 0
    constraint_violations = 0

    while state is not None:

        action = select_fcfs_action(env, state)

        if action is None:
            break

        state, reward, done, metrics = env.step(action)

        decisions += 1

        if not env.last_interval_constraint_ok:
            constraint_violations += 1

        if done:
            break

    metrics = env.get_metrics()

    metrics["episode_date"] = str(episode_date)
    metrics["allocation_decisions"] = decisions
    metrics["constraint_violations"] = constraint_violations

    return metrics


# ============================================================
# 4. MAIN EXPERIMENT
# ============================================================

def main():

    print("\nFCFS BASELINE EXPERIMENT")
    print("=" * 65)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    df["connectionTime"] = pd.to_datetime(
        df["connectionTime"],
        utc=True,
        errors="coerce"
    )

    df["disconnectTime"] = pd.to_datetime(
        df["disconnectTime"],
        utc=True,
        errors="coerce"
    )

    df = df.dropna(
        subset=["connectionTime", "disconnectTime"]
    ).copy()

    df["episode_date"] = (
        df["connectionTime"].dt.date
    )

    # --------------------------------------------------------
    # Chronological train-test date split
    # --------------------------------------------------------

    dates = sorted(
        df["episode_date"].dropna().unique()
    )

    if len(dates) < 2:
        raise ValueError(
            "At least two distinct dates are required."
        )

    split_index = int(
        len(dates) * (1 - TEST_RATIO)
    )

    split_index = max(
        1,
        min(split_index, len(dates) - 1)
    )

    train_dates = dates[:split_index]
    test_dates = dates[split_index:]

    print("\nDATASET INFORMATION")
    print("-" * 65)

    print("Total valid sessions:", len(df))
    print("Total unique dates:", len(dates))
    print("Training dates:", len(train_dates))
    print("Testing dates:", len(test_dates))

    print(
        "Training period:",
        train_dates[0],
        "to",
        train_dates[-1]
    )

    print(
        "Testing period:",
        test_dates[0],
        "to",
        test_dates[-1]
    )

    print("\nFCFS CONFIGURATION")
    print("-" * 65)

    print("Site power limit:", SITE_POWER_LIMIT, "kW")
    print("Maximum charger power:", CHARGER_POWER_KW, "kW")
    print("Power levels:", list(POWER_LEVELS))
    print("Time step:", TIME_STEP_MINUTES, "minutes")
    print("Charging spaces:", NUM_CHARGERS)

    # --------------------------------------------------------
    # Run FCFS on held-out test dates
    # --------------------------------------------------------

    results = []

    print("\nRUNNING FCFS ON TEST DATES")
    print("-" * 65)

    for count, day in enumerate(test_dates, start=1):

        day_data = df[
            df["episode_date"] == day
        ].copy()

        if day_data.empty:
            continue

        print(
            f"\n[{count}/{len(test_dates)}] "
            f"Testing date: {day} | "
            f"Sessions: {len(day_data)}"
        )

        metrics = run_fcfs_episode(
            day_data,
            day
        )

        results.append(metrics)

        print(
            "Energy fulfillment:",
            f"{metrics['energy_fulfillment_pct']:.2f}%"
        )

        print(
            "Average waiting time:",
            f"{metrics['average_waiting_time_min']:.2f} min"
        )

        print(
            "Peak site load:",
            f"{metrics['peak_site_load_kw']:.2f} kW"
        )

        print(
            "Completion rate:",
            f"{metrics['completion_rate_pct']:.2f}%"
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    if not results:
        raise RuntimeError(
            "No FCFS test results were generated."
        )

    results_df = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Aggregate test performance
    # --------------------------------------------------------

    total_sessions = results_df["total_sessions"].sum()

    total_demand = results_df["energy_demand_kwh"].sum()

    total_delivered = results_df["energy_delivered_kwh"].sum()

    total_completed = results_df["completed_sessions"].sum()

    total_shortfall = results_df["energy_shortfall_kwh"].sum()

    overall_fulfillment = (
        total_delivered / total_demand * 100
        if total_demand > 0 else 0
    )

    overall_completion = (
        total_completed / total_sessions * 100
        if total_sessions > 0 else 0
    )

    average_waiting = (
        results_df["average_waiting_time_min"]
        .mean()
    )

    peak_load = (
        results_df["peak_site_load_kw"]
        .max()
    )

    average_utilization = (
        results_df["charger_utilization_pct"]
        .mean()
    )

    total_violations = (
        results_df["constraint_violations"].sum()
    )

    total_decisions = (
        results_df["allocation_decisions"].sum()
    )

    print("\n\nFINAL FCFS TEST RESULTS")
    print("=" * 65)

    print("Test sessions:", int(total_sessions))

    print(
        "Total energy demand:",
        f"{total_demand:.2f} kWh"
    )

    print(
        "Total energy delivered:",
        f"{total_delivered:.2f} kWh"
    )

    print(
        "Energy shortfall:",
        f"{total_shortfall:.2f} kWh"
    )

    print(
        "Energy fulfillment:",
        f"{overall_fulfillment:.2f}%"
    )

    print(
        "Completed sessions:",
        int(total_completed)
    )

    print(
        "Completion rate:",
        f"{overall_completion:.2f}%"
    )

    print(
        "Average daily waiting time:",
        f"{average_waiting:.2f} min"
    )

    print(
        "Maximum peak site load:",
        f"{peak_load:.2f} kW"
    )

    print(
        "Average daily charger utilization:",
        f"{average_utilization:.2f}%"
    )

    print(
        "Allocation decisions:",
        int(total_decisions)
    )

    print(
        "Constraint violations:",
        int(total_violations)
    )

    print("\nRESULT FILE")
    print("-" * 65)
    print(OUTPUT_PATH)

    print("\nFCFS BASELINE EXPERIMENT COMPLETED")


if __name__ == "__main__":
    main()