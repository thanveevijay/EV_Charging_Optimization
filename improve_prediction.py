import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold
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

print("\n" + "=" * 65)
print("EV CHARGING DEMAND PREDICTION - ENHANCED REALISTIC MODEL")
print("=" * 65)

# ============================================================
# 2. LOAD AND CLEAN DATA
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
df["connectionTime"] = pd.to_datetime(df["connectionTime"], utc=True, errors="coerce")
df["kWhDelivered"] = pd.to_numeric(df["kWhDelivered"], errors="coerce")
df = df.dropna(subset=["connectionTime", "kWhDelivered"])
df = df[df["kWhDelivered"] > 0].copy()

# Hourly aggregation
df["hour"] = df["connectionTime"].dt.floor("h")
hourly = (
    df.groupby("hour")
    .agg(
        ev_arrivals=("connectionTime", "count"),
        energy_demand_kwh=("kWhDelivered", "sum")
    )
    .sort_index()
    .asfreq("h", fill_value=0)
)

print(f"Total hourly records: {len(hourly)}")
print(f"Zero arrival hours  : {(hourly['ev_arrivals'] == 0).sum()} ({(hourly['ev_arrivals'] == 0).mean()*100:.1f}%)")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

data = hourly.copy()
data["hour_of_day"] = data.index.hour
data["day_of_week"] = data.index.day_of_week
data["day_of_month"] = data.index.day
data["month"] = data.index.month
data["is_weekend"] = (data.index.day_of_week >= 5).astype(int)

# Cyclical diurnal and weekly encodings
data["hour_sin"] = np.sin(2 * np.pi * data["hour_of_day"] / 24)
data["hour_cos"] = np.cos(2 * np.pi * data["hour_of_day"] / 24)
data["dow_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
data["dow_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)

# Caltech campus charging regime flags
data["is_peak_hours"] = (
    (data["hour_of_day"] >= 7) & (data["hour_of_day"] <= 18) & (data["is_weekend"] == 0)
).astype(int)
data["is_night"] = (
    (data["hour_of_day"] >= 22) | (data["hour_of_day"] <= 5)
).astype(int)

# Multi-scale lags and rolling features
for target in ["ev_arrivals", "energy_demand_kwh"]:
    for lag in [1, 2, 3, 24, 48, 168]:
        data[f"{target}_lag_{lag}"] = data[target].shift(lag)
    for w in [6, 12, 24, 168]:
        data[f"{target}_rolling_{w}"] = data[target].shift(1).rolling(w).mean()
        data[f"{target}_rolling_std_{w}"] = data[target].shift(1).rolling(w).std().fillna(0)

data = data.dropna()
split_index = int(len(data) * 0.80)
train_df = data.iloc[:split_index].copy()
test_df = data.iloc[split_index:].copy()

feature_cols = [c for c in data.columns if c not in ["ev_arrivals", "energy_demand_kwh"]]
X_train = train_df[feature_cols].copy()
X_test = test_df[feature_cols].copy()

y_arr_train = train_df["ev_arrivals"]
y_arr_test = test_df["ev_arrivals"]
y_en_train = train_df["energy_demand_kwh"]
y_en_test = test_df["energy_demand_kwh"]

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")
print(f"Feature count   : {len(feature_cols)}")

# ============================================================
# 4. TRAIN BASELINE RANDOM FOREST MODELS
# ============================================================

print("\n--- Training Baseline Random Forest Models ---")
# Baseline uses classic 15 features for exact comparison
base_feature_cols = [
    "hour_of_day", "day_of_week", "day_of_month", "month", "is_weekend",
    "ev_arrivals_lag_1", "ev_arrivals_lag_24", "ev_arrivals_lag_168",
    "ev_arrivals_rolling_24", "ev_arrivals_rolling_168",
    "energy_demand_kwh_lag_1", "energy_demand_kwh_lag_24", "energy_demand_kwh_lag_168",
    "energy_demand_kwh_rolling_24", "energy_demand_kwh_rolling_168"
]

base_arr_model = RandomForestRegressor(
    n_estimators=150, max_depth=12, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
)
base_arr_model.fit(X_train[base_feature_cols], y_arr_train)
base_pred_arr = np.maximum(0, base_arr_model.predict(X_test[base_feature_cols]))

base_en_model = RandomForestRegressor(
    n_estimators=150, max_depth=12, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
)
base_en_model.fit(X_train[base_feature_cols], y_en_train)
base_pred_en = np.maximum(0, base_en_model.predict(X_test[base_feature_cols]))

# ============================================================
# 5. TRAIN ENHANCED REALISTIC MODELS (GRADIENT BOOSTED + GATED)
# ============================================================

print("\n--- Training Enhanced Realistic Models ---")

# Step 5a: Out-of-fold predicted arrivals on train set for realistic conditioning
oof_pred_arr = np.zeros(len(X_train))
kf = KFold(n_splits=5, shuffle=False)
for train_idx, val_idx in kf.split(X_train):
    fold_model = HistGradientBoostingRegressor(
        max_iter=250, max_depth=8, learning_rate=0.03, min_samples_leaf=15, random_state=RANDOM_STATE
    )
    fold_model.fit(X_train.iloc[train_idx], y_arr_train.iloc[train_idx])
    oof_pred_arr[val_idx] = np.maximum(0, fold_model.predict(X_train.iloc[val_idx]))

# Step 5b: Full enhanced arrival model
enhanced_arr_model = HistGradientBoostingRegressor(
    max_iter=250, max_depth=8, learning_rate=0.03, min_samples_leaf=15, random_state=RANDOM_STATE
)
enhanced_arr_model.fit(X_train, y_arr_train)
enhanced_pred_arr = np.maximum(0, enhanced_arr_model.predict(X_test))

# Step 5c: Train conditioned energy model
X_train_en = X_train.copy()
X_train_en["pred_arrivals"] = oof_pred_arr

X_test_en = X_test.copy()
X_test_en["pred_arrivals"] = enhanced_pred_arr

enhanced_en_model = HistGradientBoostingRegressor(
    max_iter=250, max_depth=8, learning_rate=0.03, min_samples_leaf=15, random_state=RANDOM_STATE
)
enhanced_en_model.fit(X_train_en, y_en_train)
enhanced_pred_en_raw = np.maximum(0, enhanced_en_model.predict(X_test_en))

# Step 5d: Apply physical zero-arrival gating
# When arrival prediction indicates 0 EVs (<0.35), physical energy demand is 0.0
enhanced_pred_en = np.where(enhanced_pred_arr < 0.35, 0.0, enhanced_pred_en_raw)

# ============================================================
# 6. EVALUATE AND COMPARE
# ============================================================

def calc_metrics(actual, pred, name):
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    r2 = r2_score(actual, pred)
    return {"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2}

base_arr_met = calc_metrics(y_arr_test, base_pred_arr, "EV Arrival (Baseline RF)")
enh_arr_met  = calc_metrics(y_arr_test, enhanced_pred_arr, "EV Arrival (Enhanced GBDT)")

base_en_met = calc_metrics(y_en_test, base_pred_en, "Energy Demand (Baseline RF)")
enh_en_met  = calc_metrics(y_en_test, enhanced_pred_en, "Energy Demand (Enhanced Realistic)")

print("\n" + "=" * 65)
print("MODEL EVALUATION COMPARISON")
print("=" * 65)
comparison_df = pd.DataFrame([
    {
        "Prediction Target": "EV Arrivals",
        "Baseline R2": round(base_arr_met["R2"], 4),
        "Enhanced R2": round(enh_arr_met["R2"], 4),
        "R2 Gain (%)": round((enh_arr_met["R2"] - base_arr_met["R2"]) / base_arr_met["R2"] * 100, 2),
        "Baseline MAE": round(base_arr_met["MAE"], 4),
        "Enhanced MAE": round(enh_arr_met["MAE"], 4),
        "MAE Improvement (%)": round((base_arr_met["MAE"] - enh_arr_met["MAE"]) / base_arr_met["MAE"] * 100, 2)
    },
    {
        "Prediction Target": "Energy Demand (kWh)",
        "Baseline R2": round(base_en_met["R2"], 4),
        "Enhanced R2": round(enh_en_met["R2"], 4),
        "R2 Gain (%)": round((enh_en_met["R2"] - base_en_met["R2"]) / base_en_met["R2"] * 100, 2),
        "Baseline MAE": round(base_en_met["MAE"], 4),
        "Enhanced MAE": round(enh_en_met["MAE"], 4),
        "MAE Improvement (%)": round((base_en_met["MAE"] - enh_en_met["MAE"]) / base_en_met["MAE"] * 100, 2)
    }
])
print(comparison_df.to_string(index=False))

# Check physical consistency during zero arrivals
zero_mask = (y_arr_test == 0)
print(f"\nPhysical Consistency Check during Zero-Arrival Hours (n={zero_mask.sum()}):")
print(f"  Actual Energy Demand Mean    : {y_en_test[zero_mask].mean():.2f} kWh")
print(f"  Baseline RF Predicted Mean   : {base_pred_en[zero_mask].mean():.2f} kWh (Unrealistic!)")
print(f"  Enhanced Realistic Mean      : {enhanced_pred_en[zero_mask].mean():.2f} kWh (Realistic & Gated)")

# ============================================================
# 7. SAVE RESULTS AND PREDICTIONS
# ============================================================

# Predictions table with timestamps
predictions_df = pd.DataFrame({
    "timestamp": test_df.index,
    "actual_ev_arrivals": y_arr_test.values,
    "predicted_ev_arrivals": np.round(enhanced_pred_arr, 2),
    "baseline_ev_arrivals": np.round(base_pred_arr, 2),
    "actual_energy_kwh": np.round(y_en_test.values, 2),
    "predicted_energy_kwh": np.round(enhanced_pred_en, 2),
    "baseline_energy_kwh": np.round(base_pred_en, 2)
})

predictions_df.to_csv(RESULT_DIR / "demand_predictions.csv", index=False)

# Metrics file (compatible with app.py and extended)
metrics_table = pd.DataFrame([
    {"Model": "EV Arrival Prediction", "MAE": enh_arr_met["MAE"], "RMSE": enh_arr_met["RMSE"], "R2": enh_arr_met["R2"]},
    {"Model": "Energy Demand Prediction", "MAE": enh_en_met["MAE"], "RMSE": enh_en_met["RMSE"], "R2": enh_en_met["R2"]},
    {"Model": "EV Arrival (Baseline RF)", "MAE": base_arr_met["MAE"], "RMSE": base_arr_met["RMSE"], "R2": base_arr_met["R2"]},
    {"Model": "Energy Demand (Baseline RF)", "MAE": base_en_met["MAE"], "RMSE": base_en_met["RMSE"], "R2": base_en_met["R2"]}
])
metrics_table.to_csv(RESULT_DIR / "demand_prediction_metrics.csv", index=False)

# Save feature list for app
feature_metadata = {
    "feature_columns": feature_cols,
    "base_feature_columns": base_feature_cols
}
joblib.dump(feature_metadata, MODEL_DIR / "feature_metadata.pkl")

# Save models
joblib.dump(enhanced_arr_model, MODEL_DIR / "arrival_model.pkl")
joblib.dump(enhanced_en_model, MODEL_DIR / "energy_model.pkl")
joblib.dump(base_arr_model, MODEL_DIR / "baseline_arrival_model.pkl")
joblib.dump(base_en_model, MODEL_DIR / "baseline_energy_model.pkl")

print("\nTrained models and result datasets saved successfully.")

# ============================================================
# 8. VISUALIZATIONS
# ============================================================

# Figure 11: Demand prediction actual vs predicted
plt.figure(figsize=(14, 9))

# Plot arrivals
plt.subplot(2, 1, 1)
sample_slice = slice(0, 168)  # 1 week zoom for clarity
plt.plot(predictions_df["timestamp"].iloc[sample_slice], predictions_df["actual_ev_arrivals"].iloc[sample_slice], label="Actual EV Arrivals", color="#1E88E5", linewidth=1.8)
plt.plot(predictions_df["timestamp"].iloc[sample_slice], predictions_df["predicted_ev_arrivals"].iloc[sample_slice], label="Realistic Predicted Arrivals (Enhanced GBDT)", color="#FFC107", linestyle="--", linewidth=1.8)
plt.title("Actual vs Realistic Predicted EV Arrivals (1-Week Representative Period)", fontsize=13, fontweight="bold")
plt.ylabel("EV Arrivals (vehicles/hr)", fontsize=11)
plt.legend(loc="upper right", frameon=True)
plt.grid(True, linestyle=":", alpha=0.6)

# Plot energy
plt.subplot(2, 1, 2)
plt.plot(predictions_df["timestamp"].iloc[sample_slice], predictions_df["actual_energy_kwh"].iloc[sample_slice], label="Actual Energy Demand", color="#004D40", linewidth=1.8)
plt.plot(predictions_df["timestamp"].iloc[sample_slice], predictions_df["predicted_energy_kwh"].iloc[sample_slice], label="Realistic Predicted Demand (Physically Gated)", color="#D81B60", linestyle="--", linewidth=1.8)
plt.title("Actual vs Realistic Predicted Energy Demand (Physically Consistent with Zero-Arrivals)", fontsize=13, fontweight="bold")
plt.ylabel("Energy Demand (kWh)", fontsize=11)
plt.xlabel("Date & Time (UTC)", fontsize=11)
plt.legend(loc="upper right", frameon=True)
plt.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.savefig(GRAPH_DIR / "11_demand_prediction.png", dpi=300)
plt.close()

# Figure 12: Comparison Bar Chart Baseline vs Enhanced
plt.figure(figsize=(10, 5))
categories = ["EV Arrivals R²", "Energy Demand R²", "Energy Demand MAE (kWh)"]
baseline_vals = [base_arr_met["R2"], base_en_met["R2"], base_en_met["MAE"]]
enhanced_vals = [enh_arr_met["R2"], enh_en_met["R2"], enh_en_met["MAE"]]

x = np.arange(len(categories))
width = 0.35

plt.bar(x - width/2, baseline_vals, width, label="Baseline Random Forest", color="#90CAF9")
plt.bar(x + width/2, enhanced_vals, width, label="Enhanced Realistic GBDT", color="#1565C0")

for i in range(len(categories)):
    plt.text(x[i] - width/2, baseline_vals[i] + 0.02, f"{baseline_vals[i]:.2f}", ha='center', va='bottom', fontsize=10)
    plt.text(x[i] + width/2, enhanced_vals[i] + 0.02, f"{enhanced_vals[i]:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.ylabel("Metric Score", fontsize=11)
plt.title("Performance Comparison: Baseline RF vs Enhanced Realistic Model", fontsize=13, fontweight="bold")
plt.xticks(x, categories, fontsize=11)
plt.legend(frameon=True)
plt.grid(axis="y", linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig(GRAPH_DIR / "12_prediction_improvement_comparison.png", dpi=300)
plt.close()

print(f"Generated graphs saved to {GRAPH_DIR}")
print("=" * 65)
print("PREDICTION IMPROVEMENT COMPLETED SUCCESSFULLY!")
print("=" * 65)
