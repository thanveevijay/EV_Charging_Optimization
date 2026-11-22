import os
import numpy as np
import pandas as pd

from charging_environment import EVChargingEnvironment
from dqn_agent import DQNAgent


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

DATA_PATH = "outputs/results/processed_acn_data.csv"
MODEL_PATH = "models/gnn_dqn_model.pth"

OUTPUT_PATH = "outputs/results/gnn_dqn_test_results.csv"

# NEW: Allocation log output file
ALLOCATION_LOG_PATH = (
    "outputs/results/gnn_dqn_allocation_log.csv"
)

SITE_POWER_LIMIT = 30.0
TIME_STEP = 5
CHARGER_POWER = 7.2
NUM_CHARGERS = 54


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

print("\n" + "=" * 60)
print("GNN-DQN MODEL EVALUATION")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

df["connectionTime"] = pd.to_datetime(
    df["connectionTime"], utc=True, errors="coerce"
)

df["disconnectTime"] = pd.to_datetime(
    df["disconnectTime"], utc=True, errors="coerce"
)

df = df.dropna(
    subset=["connectionTime", "disconnectTime"]
)

df["episode_date"] = df["connectionTime"].dt.date

dates = sorted(df["episode_date"].unique())

split_index = int(len(dates) * 0.70)
test_dates = dates[split_index:]

print(f"\nTotal available dates : {len(dates)}")
print(f"Training dates        : {split_index}")
print(f"Testing dates         : {len(test_dates)}")


# --------------------------------------------------
# 3. LOAD TRAINED MODEL
# --------------------------------------------------

print("\nLoading trained GNN-DQN model...")

agent = DQNAgent()
agent.load(MODEL_PATH)

print("Model loaded successfully!")
print("Exploration disabled: evaluation mode")


# --------------------------------------------------
# 4. EVALUATE EACH TEST DAY
# --------------------------------------------------

daily_results = []

# NEW: Store allocation decisions from all test days
all_allocation_logs = []

total_allocation_decisions = 0
total_constraint_violations = 0


for day_number, current_date in enumerate(test_dates, start=1):

    print(
        f"\nEvaluating day {day_number}/{len(test_dates)}: "
        f"{current_date}"
    )

    daily_data = df[
        df["episode_date"] == current_date
    ].copy()

    if daily_data.empty:
        continue

    env = EVChargingEnvironment(
        data=daily_data,
        site_power_limit=SITE_POWER_LIMIT,
        time_step=TIME_STEP,
        charger_power=CHARGER_POWER,
        num_chargers=NUM_CHARGERS
    )

    state = env.reset()

    allocation_decisions = 0

    # ----------------------------------------------
    # Run trained GNN-DQN policy
    # ----------------------------------------------

    while state is not None:

        action = agent.select_action(
            state,
            training=False
        )

        if action is None:
            break

        next_state, reward, done, info = env.step(action)

        allocation_decisions += 1

        state = next_state

        if done or state is None:
            break

    # ----------------------------------------------
    # NEW: Collect actual allocation decisions
    # ----------------------------------------------

    for allocation in env.allocation_log:

        log_entry = allocation.copy()

        # Identify the test date for this allocation
        log_entry["evaluation_date"] = str(current_date)

        all_allocation_logs.append(log_entry)

    # ----------------------------------------------
    # Collect daily metrics
    # ----------------------------------------------

    metrics = env.get_metrics()

    metrics["date"] = str(current_date)
    metrics["allocation_decisions"] = allocation_decisions

    # Use the environment's violation count if available.
    violations = metrics.get("constraint_violations", 0)

    metrics["constraint_violations"] = violations

    total_allocation_decisions += allocation_decisions
    total_constraint_violations += violations

    daily_results.append(metrics)

    print(
        f"  Sessions: {metrics.get('total_sessions', 0)}"
        f" | Fulfillment: "
        f"{metrics.get('energy_fulfillment_pct', 0):.2f}%"
        f" | Completion: "
        f"{metrics.get('completion_rate_pct', 0):.2f}%"
        f" | Allocations: {len(env.allocation_log)}"
    )


# --------------------------------------------------
# 5. SAVE DAILY RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(daily_results)

os.makedirs("outputs/results", exist_ok=True)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nDaily evaluation results saved:")
print(OUTPUT_PATH)


# --------------------------------------------------
# 6. SAVE GCN-DQN ALLOCATION LOG
# --------------------------------------------------

allocation_columns = [
    "evaluation_date",
    "interval_number",
    "sessionID",
    "historical_spaceID",
    "interval_start",
    "interval_end",
    "allocated_power_kw",
    "energy_delivered_kwh"
]

allocation_df = pd.DataFrame(
    all_allocation_logs,
    columns=allocation_columns
)

allocation_df.to_csv(
    ALLOCATION_LOG_PATH,
    index=False
)

print("\n" + "=" * 60)
print("GCN-DQN ALLOCATION LOG SAVED")
print("=" * 60)

print("Allocation records:", len(allocation_df))
print("Output file:", ALLOCATION_LOG_PATH)

if not allocation_df.empty:

    print("\nSample allocation records:")
    print(allocation_df.head(10).to_string(index=False))

else:

    print("\nWARNING: No allocation records were generated.")


# --------------------------------------------------
# 7. CALCULATE AGGREGATED TEST PERFORMANCE
# --------------------------------------------------

total_sessions = results_df["total_sessions"].sum()

total_demand = results_df["energy_demand_kwh"].sum()

total_delivered = results_df["energy_delivered_kwh"].sum()

total_shortfall = results_df["energy_shortfall_kwh"].sum()

completed_sessions = results_df["completed_sessions"].sum()

energy_fulfillment = (
    total_delivered / total_demand * 100
    if total_demand > 0 else 0
)

completion_rate = (
    completed_sessions / total_sessions * 100
    if total_sessions > 0 else 0
)

average_waiting = results_df[
    "average_waiting_time_min"
].mean()

maximum_peak_load = results_df[
    "peak_site_load_kw"
].max()

average_utilization = results_df[
    "charger_utilization_pct"
].mean()


# --------------------------------------------------
# 8. DISPLAY FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 60)
print("GNN-DQN TEST PERFORMANCE SUMMARY")
print("=" * 60)

print(f"Test dates                  : {len(results_df)}")
print(f"Test sessions               : {total_sessions}")
print(f"Total energy demand         : {total_demand:.2f} kWh")
print(f"Total energy delivered      : {total_delivered:.2f} kWh")
print(f"Energy shortfall            : {total_shortfall:.2f} kWh")
print(f"Energy fulfillment          : {energy_fulfillment:.2f}%")
print(f"Completed sessions          : {completed_sessions}")
print(f"Completion rate             : {completion_rate:.2f}%")
print(f"Average daily waiting time  : {average_waiting:.2f} min")
print(f"Maximum peak site load      : {maximum_peak_load:.2f} kW")
print(f"Average daily utilization   : {average_utilization:.2f}%")
print(f"Allocation decisions        : {total_allocation_decisions}")
print(f"Logged allocation records   : {len(allocation_df)}")
print(f"Constraint violations       : {total_constraint_violations}")

print("\n" + "=" * 60)
print("GNN-DQN EVALUATION COMPLETED")
print("=" * 60)