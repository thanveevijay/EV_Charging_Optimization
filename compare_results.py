
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------
# 1. FILE PATHS
# --------------------------------------------------

FCFS_PATH = "outputs/results/fcfs_baseline_results.csv"
GNN_PATH = "outputs/results/gnn_dqn_test_results.csv"

OUTPUT_DIR = "outputs/results"
GRAPH_DIR = "outputs/graphs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)


# --------------------------------------------------
# 2. LOAD RESULTS
# --------------------------------------------------

fcfs = pd.read_csv(FCFS_PATH)
gnn = pd.read_csv(GNN_PATH)

# Aggregate daily results consistently
def aggregate_results(df):

    total_sessions = df["total_sessions"].sum()
    total_demand = df["energy_demand_kwh"].sum()
    total_delivered = df["energy_delivered_kwh"].sum()
    total_completed = df["completed_sessions"].sum()

    return {
        "Energy Fulfillment (%)":
            total_delivered / total_demand * 100,

        "Completion Rate (%)":
            total_completed / total_sessions * 100,

        "Average Waiting Time (min)":
            df["average_waiting_time_min"].mean(),

        "Maximum Peak Load (kW)":
            df["peak_site_load_kw"].max(),

        "Average Charger Utilization (%)":
            df["charger_utilization_pct"].mean(),

        "Energy Delivered (kWh)":
            total_delivered,

        "Energy Shortfall (kWh)":
            df["energy_shortfall_kwh"].sum(),

        "Constraint Violations":
            df.get(
                "constraint_violations",
                pd.Series([0] * len(df))
            ).sum()
    }


fcfs_metrics = aggregate_results(fcfs)
gnn_metrics = aggregate_results(gnn)


# --------------------------------------------------
# 3. CREATE COMPARISON TABLE
# --------------------------------------------------

comparison = pd.DataFrame({
    "Metric": list(fcfs_metrics.keys()),
    "FCFS": list(fcfs_metrics.values()),
    "GNN-DQN": list(gnn_metrics.values())
})

comparison["Difference (GNN-DQN - FCFS)"] = (
    comparison["GNN-DQN"] - comparison["FCFS"]
)

comparison.to_csv(
    f"{OUTPUT_DIR}/algorithm_comparison.csv",
    index=False
)

print("\n" + "=" * 65)
print("FCFS VS GNN-DQN PERFORMANCE COMPARISON")
print("=" * 65)

print(comparison.to_string(index=False))


# --------------------------------------------------
# 4. PERCENTAGE IMPROVEMENT
# --------------------------------------------------

print("\n" + "=" * 65)
print("GNN-DQN PERFORMANCE CHANGE")
print("=" * 65)

higher_is_better = [
    "Energy Fulfillment (%)",
    "Completion Rate (%)",
    "Energy Delivered (kWh)",
    "Average Charger Utilization (%)"
]

lower_is_better = [
    "Average Waiting Time (min)",
    "Maximum Peak Load (kW)",
    "Energy Shortfall (kWh)",
    "Constraint Violations"
]

for metric in higher_is_better + lower_is_better:

    baseline = fcfs_metrics[metric]
    proposed = gnn_metrics[metric]

    if baseline == 0:
        print(f"{metric}: Percentage change unavailable")
        continue

    change = (proposed - baseline) / abs(baseline) * 100

    if metric in higher_is_better:
        status = "Improvement" if change > 0 else "Decrease"
    else:
        status = "Improvement" if change < 0 else "Increase"

    print(f"{metric}: {change:+.2f}% ({status})")


# --------------------------------------------------
# 5. COMPARISON GRAPHS
# --------------------------------------------------

plot_metrics = [
    "Energy Fulfillment (%)",
    "Completion Rate (%)",
    "Average Waiting Time (min)",
    "Maximum Peak Load (kW)",
    "Average Charger Utilization (%)"
]

for metric in plot_metrics:

    values = [
        fcfs_metrics[metric],
        gnn_metrics[metric]
    ]

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        ["FCFS", "GNN-DQN"],
        values
    )

    plt.title(f"FCFS vs GNN-DQN: {metric}")
    plt.ylabel(metric)
    plt.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, values):

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    filename = (
        metric.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
    )

    plt.savefig(
        f"{GRAPH_DIR}/10_comparison_{filename}.png",
        dpi=300
    )

    plt.close()


print("\nComparison CSV saved:")
print(f"{OUTPUT_DIR}/algorithm_comparison.csv")

print("\nComparison graphs saved in:")
print(GRAPH_DIR)

print("\nCOMPARISON COMPLETED SUCCESSFULLY!")