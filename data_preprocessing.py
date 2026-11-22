
import json
import pandas as pd
import numpy as np
from pathlib import Path

# ==================================================
# 1. CONFIGURATION
# ==================================================

DATA_PATH = Path("dataset/acndata_sessions_repaired.json")

OUTPUT_DIR = Path("outputs/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_PATH = OUTPUT_DIR / "processed_acn_data.csv"
OUTLIER_PATH = OUTPUT_DIR / "power_outliers.csv"

MAX_PLAUSIBLE_POWER_KW = 7.2  # Provisional review threshold

# ==================================================
# 2. LOAD DATASET
# ==================================================

print("\nLOADING ACN DATASET")
print("=" * 50)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH.resolve()}"
    )

with open(DATA_PATH, "r", encoding="utf-8") as file:
    raw_data = json.load(file)

if "_items" not in raw_data:
    raise ValueError("Expected '_items' key in ACN dataset.")

df = pd.json_normalize(raw_data["_items"])

original_count = len(df)

print("Original records:", original_count)
print("Original columns:", len(df.columns))

# ==================================================
# 3. TIMESTAMP CONVERSION
# ==================================================

print("\nTIMESTAMP VALIDATION")
print("=" * 50)

time_columns = [
    "connectionTime",
    "disconnectTime",
    "doneChargingTime"
]

for col in time_columns:
    df[col] = pd.to_datetime(
        df[col],
        format="%a, %d %b %Y %H:%M:%S GMT",
        utc=True,
        errors="coerce"
    )

    print(
        f"{col}: {df[col].notna().sum()} valid timestamps"
    )

# ==================================================
# 4. ESSENTIAL DATA VALIDATION
# ==================================================

print("\nDATA VALIDATION")
print("=" * 50)

# Convert energy to numeric
df["kWhDelivered"] = pd.to_numeric(
    df["kWhDelivered"],
    errors="coerce"
)

# Remove records missing essential information
df = df.dropna(
    subset=[
        "connectionTime",
        "disconnectTime",
        "kWhDelivered"
    ]
).copy()

# Retain positive energy sessions
df = df[df["kWhDelivered"] > 0].copy()

# Remove sessions with disconnect before connection
df = df[
    df["disconnectTime"] >= df["connectionTime"]
].copy()

print("Records after essential validation:", len(df))

# ==================================================
# 5. FEATURE ENGINEERING
# ==================================================

print("\nFEATURE ENGINEERING")
print("=" * 50)

# Arrival and departure features
df["arrival_hour"] = df["connectionTime"].dt.hour

df["departure_hour"] = df["disconnectTime"].dt.hour

df["day_of_week"] = (
    df["connectionTime"].dt.day_of_week
)

df["arrival_date"] = (
    df["connectionTime"].dt.date
)

# Charging duration in minutes
df["charging_duration_min"] = (
    df["doneChargingTime"] -
    df["connectionTime"]
).dt.total_seconds() / 60

# Parking duration in minutes
df["parking_duration_min"] = (
    df["disconnectTime"] -
    df["connectionTime"]
).dt.total_seconds() / 60

# ==================================================
# 6. HANDLE CHARGING DURATION ANOMALIES
# ==================================================

print("\nCHARGING DURATION VALIDATION")
print("=" * 50)

# Identify genuinely observed positive durations
df["charging_duration_valid"] = (
    df["charging_duration_min"].notna()
    & (df["charging_duration_min"] > 0)
)

# Identify missing charging completion timestamps
missing_done = df["doneChargingTime"].isna()

# Flag estimated durations separately
df["charging_duration_estimated"] = missing_done

# For missing completion times, use parking duration
# as a conservative proxy, not as observed charging time.
df.loc[
    missing_done,
    "charging_duration_min"
] = df.loc[
    missing_done,
    "parking_duration_min"
]

# Mark non-positive durations as unavailable
df.loc[
    df["charging_duration_min"] <= 0,
    "charging_duration_min"
] = np.nan

# Flag unavailable durations
df["charging_duration_unavailable"] = (
    df["charging_duration_min"].isna()
)

print(
    "Observed valid durations:",
    df["charging_duration_valid"].sum()
)

print(
    "Estimated durations:",
    df["charging_duration_estimated"].sum()
)

print(
    "Unavailable durations:",
    df["charging_duration_unavailable"].sum()
)

# ==================================================
# 7. CALCULATE AVERAGE CHARGING POWER
# ==================================================

print("\nPOWER CALCULATION")
print("=" * 50)

# Average power = energy / charging time in hours
df["raw_average_power_kw"] = (
    df["kWhDelivered"] /
    (df["charging_duration_min"] / 60)
)

# Replace infinite values
df["raw_average_power_kw"] = (
    df["raw_average_power_kw"]
    .replace([np.inf, -np.inf], np.nan)
)

# Identify unavailable power values
df["power_unavailable"] = (
    df["raw_average_power_kw"].isna()
)

# Flag high power values for further investigation
df["power_needs_review"] = (
    df["raw_average_power_kw"] >
    MAX_PLAUSIBLE_POWER_KW
)

# Preserve the original calculated power.
# Do not delete or clip suspicious observations.
df["average_power_kw"] = df["raw_average_power_kw"]

# ==================================================
# 8. POWER QUALITY REPORT
# ==================================================

print("\nPOWER QUALITY REPORT")
print("=" * 50)

print("Total sessions:", len(df))

print(
    "Power values available:",
    (~df["power_unavailable"]).sum()
)

print(
    "Power values requiring review:",
    df["power_needs_review"].sum()
)

print(
    "Power values unavailable:",
    df["power_unavailable"].sum()
)

print(
    "Maximum raw power:",
    round(df["raw_average_power_kw"].max(), 2),
    "kW"
)

print("\nPower statistics:")
print(
    df["average_power_kw"]
    .describe()
    .round(2)
)

# ==================================================
# 9. SAVE POWER OUTLIERS
# ==================================================

outliers = df[
    df["power_needs_review"] |
    df["power_unavailable"]
].copy()

outliers.to_csv(
    OUTLIER_PATH,
    index=False
)

print("\nPower review records saved:", OUTLIER_PATH)

# ==================================================
# 10. SELECT FINAL FEATURES
# ==================================================

selected_columns = [
    "sessionID",
    "siteID",
    "stationID",
    "spaceID",

    "connectionTime",
    "disconnectTime",
    "doneChargingTime",

    "arrival_date",
    "arrival_hour",
    "departure_hour",
    "day_of_week",

    "kWhDelivered",

    "charging_duration_min",
    "parking_duration_min",

    "average_power_kw",
    "raw_average_power_kw",

    "charging_duration_valid",
    "charging_duration_estimated",
    "charging_duration_unavailable",

    "power_needs_review",
    "power_unavailable"
]

processed_df = df[selected_columns].copy()

# ==================================================
# 11. SAVE PROCESSED DATASET
# ==================================================

processed_df.to_csv(
    PROCESSED_PATH,
    index=False
)

# ==================================================
# 12. FINAL DATA QUALITY SUMMARY
# ==================================================

print("\nFINAL PREPROCESSING RESULTS")
print("=" * 50)

print("Original records:", original_count)
print("Retained records:", len(processed_df))
print("Removed records:", original_count - len(processed_df))

print("\nCHARGING DURATION SUMMARY")

print(
    "Observed valid:",
    processed_df["charging_duration_valid"].sum()
)

print(
    "Estimated:",
    processed_df["charging_duration_estimated"].sum()
)

print(
    "Unavailable:",
    processed_df["charging_duration_unavailable"].sum()
)

print("\nPOWER SUMMARY")

print(
    "Power values requiring review:",
    processed_df["power_needs_review"].sum()
)

print(
    "Power values unavailable:",
    processed_df["power_unavailable"].sum()
)

print("\nMISSING VALUES")
print(processed_df.isnull().sum())

print("\nPROCESSED FEATURE STATISTICS")

print(
    processed_df[
        [
            "kWhDelivered",
            "charging_duration_min",
            "parking_duration_min",
            "average_power_kw"
        ]
    ].describe().round(2)
)

print("\nOUTPUT FILES")
print("=" * 50)

print("Processed dataset:", PROCESSED_PATH)
print("Power review dataset:", OUTLIER_PATH)

print("\nACN DATA PREPROCESSING COMPLETED SUCCESSFULLY!")