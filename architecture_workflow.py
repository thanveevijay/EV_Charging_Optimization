
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Create figure
fig, ax = plt.subplots(figsize=(16, 8))
ax.set_xlim(0, 16)
ax.set_ylim(0, 8)
ax.axis("off")

def draw_box(x, y, width, height, title, description, color):
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.15",
        linewidth=1.8,
        edgecolor="black",
        facecolor=color
    )
    ax.add_patch(box)

    ax.text(
        x + width / 2,
        y + height * 0.68,
        title,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold"
    )

    ax.text(
        x + width / 2,
        y + height * 0.32,
        description,
        ha="center",
        va="center",
        fontsize=9,
        wrap=True
    )

def draw_arrow(x1, y1, x2, y2):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="->",
        mutation_scale=18,
        linewidth=1.8,
        color="black"
    )
    ax.add_patch(arrow)

# Main workflow
draw_box(
    0.4, 4.8, 2.3, 1.5,
    "ACN-Data",
    "EV charging sessions\nand charging details",
    "#D6EAF8"
)

draw_box(
    3.3, 4.8, 2.3, 1.5,
    "Preprocessing",
    "Cleaning and\nfeature generation",
    "#D5F5E3"
)

draw_box(
    6.2, 4.8, 2.3, 1.5,
    "Graph Construction",
    "EV, charging point\nand station nodes",
    "#FCF3CF"
)

draw_box(
    9.1, 4.8, 2.3, 1.5,
    "GCN Encoder",
    "Graph convolution\nand state embedding",
    "#E8DAEF"
)

draw_box(
    12.0, 4.8, 2.3, 1.5,
    "DQN Scheduler",
    "State evaluation\nand action selection",
    "#FADBD8"
)

draw_arrow(2.7, 5.55, 3.3, 5.55)
draw_arrow(5.6, 5.55, 6.2, 5.55)
draw_arrow(8.5, 5.55, 9.1, 5.55)
draw_arrow(11.4, 5.55, 12.0, 5.55)

# Action selection
draw_box(
    12.0, 2.0, 2.3, 1.5,
    "Scheduling Action",
    "FCFS | Earliest Departure\nHighest Remaining Demand",
    "#FDEBD0"
)

draw_arrow(13.15, 4.8, 13.15, 3.5)

# Simulation environment
draw_box(
    8.2, 2.0, 2.8, 1.5,
    "Charging Simulator",
    "Resource allocation\nand power constraints",
    "#D6EAF8"
)

draw_arrow(12.0, 2.75, 11.0, 2.75)

# Reward and feedback
draw_box(
    3.8, 2.0, 3.0, 1.5,
    "Reward Calculation",
    "Waiting time, peak load\nand energy fulfillment",
    "#D5F5E3"
)

draw_arrow(8.2, 2.75, 6.8, 2.75)

# Feedback loop
draw_arrow(3.8, 2.75, 1.55, 2.75)
ax.plot([1.55, 1.55], [2.75, 4.8],
        color="black", linewidth=1.5)
ax.annotate(
    "",
    xy=(1.55, 4.8),
    xytext=(1.55, 4.35),
    arrowprops=dict(arrowstyle="->", linewidth=1.5)
)

ax.text(
    2.0, 3.55,
    "Feedback for DQN training",
    fontsize=9,
    fontweight="bold"
)

# Title
ax.set_title(
    "GCN-DQN Based Dynamic EV Charging Optimization Workflow",
    fontsize=16,
    fontweight="bold",
    pad=20
)

plt.tight_layout()

# Save figure
os.makedirs("outputs/graphs", exist_ok=True)

output_path = "outputs/graphs/figure_5_4_gcn_dqn_workflow.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

print("GCN-DQN WORKFLOW DIAGRAM GENERATED")
print("-----------------------------------")
print("1. ACN-Data input")
print("2. Data preprocessing")
print("3. Graph construction")
print("4. GCN state representation")
print("5. DQN scheduling action")
print("6. Charging simulation")
print("7. Reward calculation and feedback")
print("-----------------------------------")
print("Figure saved at:", output_path)

plt.show()