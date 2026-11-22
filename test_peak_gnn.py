
import pandas as pd
import numpy as np
import torch

from gnn_model import (
    EVChargingGNN,
    build_charging_graph,
    POWER_LEVELS,
    NUM_POWER_LEVELS
)

DATA_PATH = "outputs/results/processed_acn_data.csv"

PEAK_TIME = pd.Timestamp("2018-09-24 21:37:19", tz="UTC")
SITE_POWER_LIMIT = 30.0
CHARGER_POWER = 7.2

print("\nTESTING GNN WITH HISTORICAL PEAK EV ACTIVITY")
print("=" * 60)

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

for col in ["connectionTime", "disconnectTime"]:
    df[col] = pd.to_datetime(
        df[col], utc=True, errors="coerce"
    )

df["kWhDelivered"] = pd.to_numeric(
    df["kWhDelivered"], errors="coerce"
)

df = df.dropna(
    subset=["connectionTime", "disconnectTime", "kWhDelivered"]
)

# --------------------------------------------------
# 2. FIND ACTIVE EVs AT PEAK TIME
# --------------------------------------------------

active = df[
    (df["connectionTime"] <= PEAK_TIME) &
    (df["disconnectTime"] > PEAK_TIME)
].copy()

active = active.reset_index(drop=True)

print("\nHISTORICAL PEAK SNAPSHOT")
print("-" * 60)
print("Peak timestamp:", PEAK_TIME)
print("Active EVs:", len(active))
print("Infrastructure spaces:", 54)

if len(active) == 0:
    raise ValueError("No active EVs found at the selected timestamp.")

# --------------------------------------------------
# 3. CHECK PHYSICAL SPACE OCCUPANCY
# --------------------------------------------------

duplicate_spaces = active[
    active["spaceID"].duplicated(keep=False)
]

print("\nPHYSICAL SPACE VALIDATION")
print("-" * 60)

if duplicate_spaces.empty:
    print("Duplicate occupied spaces: 0")
    print("Space occupancy validation: PASSED")
else:
    print("Duplicate occupied spaces detected:")
    print(
        duplicate_spaces[
            ["sessionID", "spaceID", "connectionTime", "disconnectTime"]
        ].to_string(index=False)
    )
    raise ValueError(
        "Multiple active sessions share a physical space. "
        "Resolve this before running the simulator."
    )

# --------------------------------------------------
# 4. CREATE EIGHT FEATURES FOR EACH EV
# --------------------------------------------------

ev_features = []

for _, row in active.iterrows():

    energy = max(float(row["kWhDelivered"]), 0.0)

    remaining_hours = max(
        (row["disconnectTime"] - PEAK_TIME).total_seconds()
        / 3600,
        1 / 60
    )

    waiting_hours = max(
        (PEAK_TIME - row["connectionTime"]).total_seconds()
        / 3600,
        0
    )

    parking_hours = max(
        (row["disconnectTime"] - row["connectionTime"]).total_seconds()
        / 3600,
        1 / 60
    )

    urgency = min(
        energy / (CHARGER_POWER * remaining_hours),
        1.0
    )

    estimated_average_power = energy / parking_hours

    features = [
        energy / 30.0,
        remaining_hours / 12.0,
        urgency,
        energy / 30.0,
        0.0,
        1.0,
        estimated_average_power / CHARGER_POWER,
        waiting_hours / 12.0
    ]

    ev_features.append(features)

ev_features = np.asarray(ev_features, dtype=np.float32)

# --------------------------------------------------
# 5. CREATE SITE FEATURES
# --------------------------------------------------

site_features = np.array([
    SITE_POWER_LIMIT / 60.0,
    SITE_POWER_LIMIT / 60.0,
    0.0,
    len(active) / 54.0,
    PEAK_TIME.hour / 24.0,
    PEAK_TIME.dayofweek / 6.0,
    len(active) / 54.0,
    0.0
], dtype=np.float32)

# Each EV can individually receive one of the three
# charger power levels. Aggregate budget is handled
# dynamically as allocations are made.
action_mask = np.ones(
    (len(active), NUM_POWER_LEVELS),
    dtype=bool
)

# --------------------------------------------------
# 6. BUILD GRAPH
# --------------------------------------------------

node_features, edge_index, action_mask = build_charging_graph(
    site_features,
    ev_features,
    action_mask
)

print("\nGRAPH VALIDATION")
print("-" * 60)
print("EV nodes:", len(active))
print("Total graph nodes:", node_features.shape[0])
print("Graph edges:", edge_index.shape[1])
print("EV feature shape:", ev_features.shape)
print("Action mask shape:", action_mask.shape)
print("Available power levels:", POWER_LEVELS)

# --------------------------------------------------
# 7. RUN GNN
# --------------------------------------------------

model = EVChargingGNN()
model.eval()

with torch.no_grad():
    q_values, embeddings = model(
        node_features,
        edge_index,
        action_mask
    )

print("\nGNN OUTPUT")
print("-" * 60)
print("EV embeddings:", embeddings[1:].shape)
print("Q-value matrix:", q_values.shape)
print("Finite Q-values:", torch.isfinite(q_values).all().item())

# --------------------------------------------------
# 8. TEST POWER BUDGET
# --------------------------------------------------

print("\nPOWER BUDGET CHECK")
print("-" * 60)

maximum_possible_demand = len(active) * CHARGER_POWER

print("Site power limit:", SITE_POWER_LIMIT, "kW")
print("Maximum individual charger power:", CHARGER_POWER, "kW")
print("Unrestricted combined charger demand:",
      round(maximum_possible_demand, 2), "kW")

print("Required aggregate allocation limit:",
      SITE_POWER_LIMIT, "kW")

# --------------------------------------------------
# 9. FINAL VALIDATION
# --------------------------------------------------

assert node_features.shape[0] == len(active) + 1
assert q_values.shape == (len(active), NUM_POWER_LEVELS)
assert torch.isfinite(q_values).all()

print("\nPEAK GNN TEST: PASSED")
