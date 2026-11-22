
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --------------------------------------------------
# PATHS
# --------------------------------------------------
INPUT_FILE = "outputs/results/gnn_dqn_allocation_log.csv"
OUTPUT_DIR = "outputs/graphs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# LOAD ALLOCATION LOG
# --------------------------------------------------
df = pd.read_csv(INPUT_FILE)

df["interval_start"] = pd.to_datetime(
    df["interval_start"], errors="coerce", utc=True
)
df["interval_end"] = pd.to_datetime(
    df["interval_end"], errors="coerce", utc=True
)

df = df.dropna(
    subset=["interval_start", "interval_end", "sessionID",
            "allocated_power_kw"]
).copy()

df["allocated_power_kw"] = pd.to_numeric(
    df["allocated_power_kw"], errors="coerce"
)
df = df.dropna(subset=["allocated_power_kw"])

# Select the test day with the most allocation records
selected_date = df["evaluation_date"].value_counts().idxmax()
day = df[df["evaluation_date"] == selected_date].copy()

print("Selected evaluation date:", selected_date)
print("Allocation records:", len(day))

# --------------------------------------------------
# FIGURE 6.2: EV POWER ALLOCATION TIMELINE
# --------------------------------------------------

# Select 20 EV sessions with the most allocation intervals
top_sessions = (
    day.groupby("sessionID")
    .size()
    .sort_values(ascending=False)
    .head(20)
    .index
)

timeline = day[day["sessionID"].isin(top_sessions)].copy()

session_order = list(top_sessions)
session_positions = {
    session: i for i, session in enumerate(session_order)
}

fig, ax = plt.subplots(figsize=(14, 8))

power_colors = {
    2.4: "steelblue",
    4.8: "darkorange",
    7.2: "seagreen"
}

for _, row in timeline.iterrows():
    session = row["sessionID"]

    start = row["interval_start"]
    end = row["interval_end"]

    start_num = mdates.date2num(start.to_pydatetime())
    end_num = mdates.date2num(end.to_pydatetime())

    duration = end_num - start_num

    power = float(row["allocated_power_kw"])
    color = min(
        power_colors,
        key=lambda p: abs(p - power)
    )

    ax.broken_barh(
        [(start_num, duration)],
        (session_positions[session] - 0.35, 0.7),
        facecolors=power_colors[color],
        edgecolors="black",
        linewidth=0.2
    )

ax.set_yticks(range(len(session_order)))
ax.set_yticklabels([str(s) for s in session_order], fontsize=8)

ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=mdates.UTC))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))

ax.set_xlabel("Time of Day (UTC)")
ax.set_ylabel("EV Session ID")
ax.set_title(
    "Dynamic EV Charging Power Allocation Timeline (GCN-DQN)"
)

ax.grid(axis="x", linestyle="--", alpha=0.5)

# Legend
handles = [
    plt.Rectangle((0, 0), 1, 1, color=power_colors[p])
    for p in [2.4, 4.8, 7.2]
]

ax.legend(
    handles,
    ["2.4 kW", "4.8 kW", "7.2 kW"],
    title="Allocated Power"
)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "figure_6_2_dynamic_power_allocation.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# --------------------------------------------------
# FIGURE 6.3: CHARGING STATION POWER-LOAD PROFILE
# --------------------------------------------------

# Sum the allocated power across EVs at each interval
load_profile = (
    day.groupby("interval_start")["allocated_power_kw"]
    .sum()
    .sort_index()
)

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(
    load_profile.index,
    load_profile.values,
    linewidth=1.8,
    label="GCN-DQN Station Load"
)

ax.axhline(
    y=30,
    color="red",
    linestyle="--",
    linewidth=1.5,
    label="Site Power Limit (30 kW)"
)

ax.set_xlabel("Time")
ax.set_ylabel("Total Allocated Power (kW)")
ax.set_title("Charging Station Power-Load Profile (GCN-DQN)")

ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%H:%M", tz=mdates.UTC)
)
ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))

ax.grid(True, linestyle="--", alpha=0.5)
ax.legend()

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "figure_6_3_station_power_load.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()

print("\nBoth figures generated successfully!")
print("Figure 6.2: figure_6_2_dynamic_power_allocation.png")
print("Figure 6.3: figure_6_3_station_power_load.png")
print("Maximum plotted load:", round(load_profile.max(), 2), "kW")