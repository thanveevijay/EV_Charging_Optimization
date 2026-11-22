
import pandas as pd

DATA_PATH = "outputs/results/processed_acn_data.csv"

print("\nANALYSING SIMULTANEOUS EV ACTIVITY")
print("=" * 55)

df = pd.read_csv(DATA_PATH)

df["connectionTime"] = pd.to_datetime(
    df["connectionTime"], utc=True, errors="coerce"
)

df["disconnectTime"] = pd.to_datetime(
    df["disconnectTime"], utc=True, errors="coerce"
)

df = df.dropna(subset=["connectionTime", "disconnectTime"])
df = df[df["disconnectTime"] > df["connectionTime"]].copy()

# Examine each calendar day independently.
df["date"] = df["connectionTime"].dt.date

daily_results = []

for date, day_df in df.groupby("date"):
    events = []

    for _, row in day_df.iterrows():
        events.append((row["connectionTime"], 1))
        events.append((row["disconnectTime"], -1))

    # Process departures before arrivals at identical timestamps.
    events.sort(key=lambda x: (x[0], x[1]))

    active = 0
    peak = 0
    peak_time = None

    for timestamp, change in events:
        active += change

        if active > peak:
            peak = active
            peak_time = timestamp

    daily_results.append({
        "date": date,
        "sessions": len(day_df),
        "peak_active_evs": peak,
        "peak_time": peak_time
    })

results = pd.DataFrame(daily_results)
results = results.sort_values(
    "peak_active_evs", ascending=False
)

print("\nTOP 10 BUSIEST DAYS")
print("-" * 55)
print(results.head(10).to_string(index=False))

best_day = results.iloc[0]

print("\nSELECTED PEAK ACTIVITY PERIOD")
print("-" * 55)
print("Date:", best_day["date"])
print("Sessions:", best_day["sessions"])
print("Maximum simultaneous EVs:", best_day["peak_active_evs"])
print("Peak timestamp:", best_day["peak_time"])

print("\nANALYSIS COMPLETED")
