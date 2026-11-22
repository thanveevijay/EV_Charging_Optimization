
import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("outputs/results/simulation_sessions.csv")

df["arrival_time"] = pd.to_datetime(df["arrival_time"], utc=True)

# Calculate actual charging start and end
df["charging_start"] = (
    df["arrival_time"] +
    pd.to_timedelta(df["waiting_time_min"], unit="m")
)

df["charging_end"] = (
    df["charging_start"] +
    pd.to_timedelta(df["charging_time_min"], unit="m")
)

# Select the first simulation date
df["date"] = df["arrival_time"].dt.date
first_day = df["date"].min()

sample = df[df["date"] == first_day].copy()
sample = sample.sort_values("charging_start").head(20)

# Convert timestamps to hours from midnight
midnight = pd.Timestamp(str(first_day), tz="UTC")

sample["start_hour"] = (
    sample["charging_start"] - midnight
).dt.total_seconds() / 3600

sample["end_hour"] = (
    sample["charging_end"] - midnight
).dt.total_seconds() / 3600

# Plot
fig, ax = plt.subplots(figsize=(12, 7))

for i, (_, row) in enumerate(sample.iterrows()):
    ax.barh(
        i,
        row["end_hour"] - row["start_hour"],
        left=row["start_hour"],
        height=0.6
    )

ax.set_yticks(range(len(sample)))
ax.set_yticklabels([
    f"EV {i+1}" for i in range(len(sample))
])

ax.set_xlabel("Time of Day (Hours)")
ax.set_ylabel("EV Sessions")
ax.set_title("FCFS Scheduling Timeline")

ax.set_xlim(0, 24)
ax.set_xticks(range(0, 25, 2))
ax.grid(axis="x", linestyle="--", alpha=0.5)
ax.invert_yaxis()

plt.tight_layout()

os.makedirs("outputs/graphs", exist_ok=True)
plt.savefig(
    "outputs/graphs/figure_6_1_fcfs_timeline.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Figure 6.1 generated.")
print("Simulation date:", first_day)
print("Sessions plotted:", len(sample))
