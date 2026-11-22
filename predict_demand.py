
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = Path("outputs/results/processed_acn_data.csv")

MODEL_DIR = Path("models/demand_prediction")
RESULT_DIR = Path("outputs/results")
GRAPH_DIR = Path("outputs/graphs")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("EV CHARGING DEMAND PREDICTION")
print("=" * 60)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Run data_preprocessing.py first."
    )

df = pd.read_csv(DATA_PATH)

print("\nOriginal dataset shape:", df.shape)
print("Available columns:", df.columns.tolist())

required_columns = ["connectionTime", "kWhDelivered"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")

print("\nLoaded records:", len(df))


# ============================================================
# 3. TIMESTAMP AND ENERGY CLEANING
# ============================================================

# Automatically recognize different timestamp formats.
df["connectionTime"] = pd.to_datetime(
    df["connectionTime"],
    utc=True,
    errors="coerce"
)

df["kWhDelivered"] = pd.to_numeric(
    df["kWhDelivered"],
    errors="coerce"
)

print("\nTimestamp parsing check:")
print(
    "Valid timestamps:",
    df["connectionTime"].notna().sum()
)
print(
    "Missing timestamps:",
    df["connectionTime"].isna().sum()
)

print("\nSample timestamps:")
print(df["connectionTime"].head())

# Remove invalid records.
df = df.dropna(
    subset=["connectionTime", "kWhDelivered"]
)

df = df[df["kWhDelivered"] > 0].copy()

if df.empty:
    raise ValueError(
        "No valid sessions remain after cleaning. "
        "Check the timestamp and energy columns."
    )

print("\nValid sessions after cleaning:", len(df))


# ============================================================
# 4. CREATE HOURLY DEMAND DATA
# ============================================================

df["hour"] = df["connectionTime"].dt.floor("h")

hourly = (
    df.groupby("hour")
    .agg(
        ev_arrivals=("connectionTime", "count"),
        energy_demand_kwh=("kWhDelivered", "sum")
    )
    .sort_index()
)

# Include missing hours with zero arrivals and demand.
hourly = hourly.asfreq("h", fill_value=0)

print("\nHourly dataset:")
print("Total hourly records:", len(hourly))
print(hourly.head())


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================

data = hourly.copy()

# Calendar features.
data["hour_of_day"] = data.index.hour
data["day_of_week"] = data.index.day_of_week
data["day_of_month"] = data.index.day
data["month"] = data.index.month

data["is_weekend"] = (
    data.index.day_of_week >= 5
).astype(int)

# Historical demand features.
target_columns = [
    "ev_arrivals",
    "energy_demand_kwh"
]

for target in target_columns:

    data[f"{target}_lag_1"] = (
        data[target].shift(1)
    )

    data[f"{target}_lag_24"] = (
        data[target].shift(24)
    )

    data[f"{target}_lag_168"] = (
        data[target].shift(168)
    )

    data[f"{target}_rolling_24"] = (
        data[target]
        .shift(1)
        .rolling(24)
        .mean()
    )

    data[f"{target}_rolling_168"] = (
        data[target]
        .shift(1)
        .rolling(168)
        .mean()
    )

data = data.dropna()

if len(data) < 20:
    raise ValueError(
        f"Insufficient hourly samples after feature engineering: "
        f"{len(data)}. More historical data is required."
    )

feature_columns = [
    "hour_of_day",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",

    "ev_arrivals_lag_1",
    "ev_arrivals_lag_24",
    "ev_arrivals_lag_168",
    "ev_arrivals_rolling_24",
    "ev_arrivals_rolling_168",

    "energy_demand_kwh_lag_1",
    "energy_demand_kwh_lag_24",
    "energy_demand_kwh_lag_168",
    "energy_demand_kwh_rolling_24",
    "energy_demand_kwh_rolling_168"
]

X = data[feature_columns]

y_arrivals = data["ev_arrivals"]
y_energy = data["energy_demand_kwh"]


# ============================================================
# 6. CHRONOLOGICAL TRAIN-TEST SPLIT
# ============================================================

split_index = int(len(data) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_arrivals_train = y_arrivals.iloc[:split_index]
y_arrivals_test = y_arrivals.iloc[split_index:]

y_energy_train = y_energy.iloc[:split_index]
y_energy_test = y_energy.iloc[split_index:]

if len(X_train) == 0 or len(X_test) == 0:
    raise ValueError("Training or testing dataset is empty.")

print("\nDataset split:")
print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))
print("Split type      : Chronological 80:20")


# ============================================================
# 7. TRAIN RANDOM FOREST MODELS
# ============================================================

print("\nTraining EV arrival prediction model...")

arrival_model = RandomForestRegressor(
    n_estimators=150,
    max_depth=12,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

arrival_model.fit(
    X_train,
    y_arrivals_train
)

print("Arrival prediction model trained.")

print("\nTraining energy demand prediction model...")

energy_model = RandomForestRegressor(
    n_estimators=150,
    max_depth=12,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

energy_model.fit(
    X_train,
    y_energy_train
)

print("Energy demand prediction model trained.")


# ============================================================
# 8. GENERATE TEST PREDICTIONS
# ============================================================

predicted_arrivals = np.maximum(
    0,
    arrival_model.predict(X_test)
)

predicted_energy = np.maximum(
    0,
    energy_model.predict(X_test)
)


# ============================================================
# 9. MODEL EVALUATION
# ============================================================

def evaluate_model(actual, predicted, model_name):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    r2 = r2_score(
        actual,
        predicted
    )

    print("\n" + model_name)
    print("-" * 40)
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


arrival_metrics = evaluate_model(
    y_arrivals_test,
    predicted_arrivals,
    "EV Arrival Prediction"
)

energy_metrics = evaluate_model(
    y_energy_test,
    predicted_energy,
    "Energy Demand Prediction"
)


# ============================================================
# 10. SAVE PREDICTIONS AND METRICS
# ============================================================

predictions = pd.DataFrame({
    "timestamp": X_test.index,
    "actual_ev_arrivals": y_arrivals_test.values,
    "predicted_ev_arrivals": predicted_arrivals,
    "actual_energy_kwh": y_energy_test.values,
    "predicted_energy_kwh": predicted_energy
})

predictions.to_csv(
    RESULT_DIR / "demand_predictions.csv",
    index=False
)

metrics = pd.DataFrame([
    arrival_metrics,
    energy_metrics
])

metrics.to_csv(
    RESULT_DIR / "demand_prediction_metrics.csv",
    index=False
)


# ============================================================
# 11. SAVE TRAINED MODELS
# ============================================================

joblib.dump(
    arrival_model,
    MODEL_DIR / "arrival_model.pkl"
)

joblib.dump(
    energy_model,
    MODEL_DIR / "energy_model.pkl"
)

print("\nTrained models saved successfully.")


# ============================================================
# 12. VISUALIZE ACTUAL VS PREDICTED VALUES
# ============================================================

fig, axes = plt.subplots(
    2,
    1,
    figsize=(14, 9)
)

axes[0].plot(
    predictions["timestamp"],
    predictions["actual_ev_arrivals"],
    label="Actual arrivals",
    linewidth=1
)

axes[0].plot(
    predictions["timestamp"],
    predictions["predicted_ev_arrivals"],
    label="Predicted arrivals",
    linewidth=1
)

axes[0].set_title(
    "Actual vs Predicted EV Arrivals"
)

axes[0].set_ylabel("Number of EVs")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(
    predictions["timestamp"],
    predictions["actual_energy_kwh"],
    label="Actual energy demand",
    linewidth=1
)

axes[1].plot(
    predictions["timestamp"],
    predictions["predicted_energy_kwh"],
    label="Predicted energy demand",
    linewidth=1
)

axes[1].set_title(
    "Actual vs Predicted Energy Demand"
)

axes[1].set_ylabel("Energy (kWh)")
axes[1].set_xlabel("Time")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()

plt.savefig(
    GRAPH_DIR / "11_demand_prediction.png",
    dpi=300
)

plt.show()


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DEMAND PREDICTION COMPLETED")
print("=" * 60)

print("\nTotal hourly records:", len(hourly))
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nSaved files:")
print("- models/demand_prediction/arrival_model.pkl")
print("- models/demand_prediction/energy_model.pkl")
print("- outputs/results/demand_predictions.csv")
print("- outputs/results/demand_prediction_metrics.csv")
print("- outputs/graphs/11_demand_prediction.png")

print("\nPrediction model training completed successfully.")
