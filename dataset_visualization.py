
import os
import pandas as pd
import matplotlib.pyplot as plt

# Load preprocessed dataset
df = pd.read_csv("outputs/results/processed_acn_data.csv")

# Create output directory
os.makedirs("outputs/graphs", exist_ok=True)

# Create two visualizations
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. EV charging session arrivals by hour
arrival_counts = df["arrival_hour"].value_counts().sort_index()

axes[0].bar(arrival_counts.index, arrival_counts.values)
axes[0].set_title("EV Charging Sessions by Arrival Hour")
axes[0].set_xlabel("Arrival Hour")
axes[0].set_ylabel("Number of Sessions")
axes[0].set_xticks(range(24))
axes[0].grid(axis="y", alpha=0.3)

# 2. Distribution of delivered energy
axes[1].hist(df["kWhDelivered"], bins=30, edgecolor="black")
axes[1].set_title("Distribution of Delivered Energy")
axes[1].set_xlabel("Energy Delivered (kWh)")
axes[1].set_ylabel("Number of Sessions")
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()

# Save figure
output_path = "outputs/graphs/figure_3_1_dataset_overview.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")

print("DATASET VISUALIZATION GENERATED SUCCESSFULLY")
print("Total sessions:", len(df))
print("Peak arrival hour:", arrival_counts.idxmax())
print("Sessions during peak hour:", arrival_counts.max())
print("Average delivered energy:", round(df["kWhDelivered"].mean(), 2), "kWh")
print("Figure saved at:", output_path)

plt.show()
