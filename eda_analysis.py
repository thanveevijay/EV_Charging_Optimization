
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==================================================
# 1. CONFIGURATION
# ==================================================

DATA_PATH = Path("outputs/results/processed_acn_data.csv")
GRAPH_DIR = Path("outputs/graphs")

GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================
# 2. LOAD PROCESSED DATA
# ==================================================

print("\nLOADING PROCESSED ACN DATA")
print("=" * 50)

df = pd.read_csv(DATA_PATH)

df["connectionTime"] = pd.to_datetime(
    df["connectionTime"],
    utc=True
)

df["disconnectTime"] = pd.to_datetime(
    df["disconnectTime"],
    utc=True
)

print("Total sessions:", len(df))
print("Total sites:", df["siteID"].nunique())
print("Total stations:", df["stationID"].nunique())
print("Total charging spaces:", df["spaceID"].nunique())

# ==================================================
# 3. GRAPH 1: EV ARRIVALS BY HOUR
# ==================================================

print("\nGenerating EV arrival graph...")

arrival_counts = (
    df["arrival_hour"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(10, 5))

plt.bar(
    arrival_counts.index,
    arrival_counts.values
)

plt.title("EV Charging Session Arrivals by Hour (UTC)")
plt.xlabel("Hour of Day")
plt.ylabel("Number of EV Arrivals")
plt.xticks(range(24))
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(GRAPH_DIR / "01_arrivals_by_hour.png", dpi=300)
plt.close()

# ==================================================
# 4. GRAPH 2: EV DEPARTURES BY HOUR
# ==================================================

print("Generating EV departure graph...")

departure_counts = (
    df["departure_hour"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(10, 5))

plt.bar(
    departure_counts.index,
    departure_counts.values
)

plt.title("EV Charging Session Departures by Hour (UTC)")
plt.xlabel("Hour of Day")
plt.ylabel("Number of EV Departures")
plt.xticks(range(24))
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(GRAPH_DIR / "02_departures_by_hour.png", dpi=300)
plt.close()

# ==================================================
# 5. GRAPH 3: ENERGY DISTRIBUTION
# ==================================================

print("Generating energy distribution graph...")

plt.figure(figsize=(10, 5))

plt.hist(
    df["kWhDelivered"].dropna(),
    bins=40,
    edgecolor="black"
)

plt.title("Distribution of Energy Delivered per EV Session")
plt.xlabel("Energy Delivered (kWh)")
plt.ylabel("Number of Sessions")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(GRAPH_DIR / "03_energy_distribution.png", dpi=300)
plt.close()

# ==================================================
# 6. GRAPH 4: CHARGING DURATION DISTRIBUTION
# ==================================================

print("Generating charging duration graph...")

# Exclude unavailable and estimated durations for
# the observed-duration distribution.
valid_durations = df.loc[
    df["charging_duration_valid"].astype(bool),
    "charging_duration_min"
].dropna()

# Limit the display range to improve readability.
# This is a plotting filter, not deletion from the dataset.
duration_plot = valid_durations[
    valid_durations <= 1000
]

plt.figure(figsize=(10, 5))

plt.hist(
    duration_plot,
    bins=40,
    edgecolor="black"
)

plt.title("Distribution of Observed Charging Duration")
plt.xlabel("Charging Duration (minutes)")
plt.ylabel("Number of Sessions")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(GRAPH_DIR / "04_charging_duration.png", dpi=300)
plt.close()

# ==================================================
# 7. GRAPH 5: ENERGY BY WEEKDAY
# ==================================================

print("Generating weekday energy graph...")

weekday_energy = (
    df.groupby("day_of_week")["kWhDelivered"]
    .sum()
    .reindex(range(7), fill_value=0)
)

weekday_names = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

plt.figure(figsize=(10, 5))

plt.bar(
    weekday_names,
    weekday_energy.values
)

plt.title("Total EV Charging Energy by Weekday")
plt.xlabel("Day of Week")
plt.ylabel("Total Energy Delivered (kWh)")
plt.xticks(rotation=30)
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(GRAPH_DIR / "05_energy_by_weekday.png", dpi=300)
plt.close()

# ==================================================
# 8. GRAPH 6: TOP 15 CHARGING SPACES
# ==================================================

print("Generating charging space usage graph...")

space_counts = (
    df["spaceID"]
    .value_counts()
    .head(15)
    .sort_values()
)

plt.figure(figsize=(11, 6))

plt.barh(
    space_counts.index.astype(str),
    space_counts.values
)

plt.title("Top 15 Most Frequently Used Charging Spaces")
plt.xlabel("Number of Charging Sessions")
plt.ylabel("Charging Space ID")
plt.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(GRAPH_DIR / "06_top_charging_spaces.png", dpi=300)
plt.close()

# ==================================================
# 9. ADDITIONAL SUMMARY STATISTICS
# ==================================================

print("\nEDA SUMMARY")
print("=" * 50)

print("Total sessions:", len(df))

print(
    "Average energy delivered:",
    round(df["kWhDelivered"].mean(), 2),
    "kWh"
)

print(
    "Average observed charging duration:",
    round(
        df.loc[
            df["charging_duration_valid"].astype(bool),
            "charging_duration_min"
        ].mean(),
        2
    ),
    "minutes"
)

print(
    "Peak arrival hour (UTC):",
    int(arrival_counts.idxmax())
)

print(
    "Peak arrival count:",
    int(arrival_counts.max())
)

print(
    "Unique charging spaces:",
    df["spaceID"].nunique()
)

print(
    "Unique stations:",
    df["stationID"].nunique()
)

print(
    "Unique sites:",
    df["siteID"].nunique()
)

print(
    "Power values flagged for review:",
    int(df["power_needs_review"].sum())
)

# ==================================================
# 10. SAVE SUMMARY REPORT
# ==================================================

summary = {
    "total_sessions": len(df),
    "total_sites": df["siteID"].nunique(),
    "total_stations": df["stationID"].nunique(),
    "total_charging_spaces": df["spaceID"].nunique(),
    "average_energy_kwh": df["kWhDelivered"].mean(),
    "average_observed_charging_duration_min": (
        df.loc[
            df["charging_duration_valid"].astype(bool),
            "charging_duration_min"
        ].mean()
    ),
    "peak_arrival_hour_utc": arrival_counts.idxmax(),
    "peak_arrival_count": arrival_counts.max(),
    "power_values_flagged": df["power_needs_review"].sum()
}

pd.Series(summary).to_csv(
    GRAPH_DIR / "eda_summary.csv",
    header=["Value"]
)

# ==================================================
# 11. FINAL OUTPUT
# ==================================================

print("\nEDA COMPLETED SUCCESSFULLY!")
print("=" * 50)

print("Graphs saved in:", GRAPH_DIR)

for graph in sorted(GRAPH_DIR.glob("*.png")):
    print("-", graph.name)

print("\nSummary saved:", GRAPH_DIR / "eda_summary.csv")