
import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load preprocessed dataset
df = pd.read_csv("outputs/results/processed_acn_data.csv")

# Select 12 actual charging sessions
sample = df.dropna(
    subset=["sessionID", "spaceID", "stationID"]
).head(12).copy()

# Create graph
G = nx.Graph()

for i, (_, row) in enumerate(sample.iterrows(), start=1):

    ev = f"EV-{i}"
    charger = f"CP-{i}"
    station = "ST-" + str(row["stationID"])

    G.add_node(ev, layer=0, node_type="EV Session")
    G.add_node(charger, layer=1, node_type="Charging Point")
    G.add_node(station, layer=2, node_type="Station")

    G.add_edge(ev, charger)
    G.add_edge(charger, station)

# Separate nodes by layer
ev_nodes = [n for n, d in G.nodes(data=True) if d["layer"] == 0]
charger_nodes = [n for n, d in G.nodes(data=True) if d["layer"] == 1]
station_nodes = [n for n, d in G.nodes(data=True) if d["layer"] == 2]

# Custom vertical layout with wider spacing
pos = {}

for i, node in enumerate(ev_nodes):
    pos[node] = (i * 2.5, 3)

for i, node in enumerate(charger_nodes):
    pos[node] = (i * 2.5, 1.5)

# Arrange unique stations separately
unique_stations = list(dict.fromkeys(station_nodes))

for i, node in enumerate(unique_stations):
    pos[node] = (i * 3.0, 0)

# Draw graph
fig, ax = plt.subplots(figsize=(18, 10))

nx.draw_networkx_edges(
    G, pos,
    ax=ax,
    edge_color="gray",
    width=1.5,
    alpha=0.55
)

nx.draw_networkx_nodes(
    G, pos,
    nodelist=ev_nodes,
    node_color="skyblue",
    node_size=1500,
    ax=ax,
    label="EV Sessions"
)

nx.draw_networkx_nodes(
    G, pos,
    nodelist=charger_nodes,
    node_color="orange",
    node_size=1500,
    ax=ax,
    label="Charging Points"
)

nx.draw_networkx_nodes(
    G, pos,
    nodelist=unique_stations,
    node_color="lightgreen",
    node_size=1800,
    ax=ax,
    label="Stations"
)

# Short labels prevent text overlap
nx.draw_networkx_labels(
    G, pos,
    font_size=9,
    font_weight="bold",
    ax=ax
)

ax.set_title(
    "EV Charging Infrastructure Graph Representation",
    fontsize=17,
    pad=25
)

ax.text(-1, 3, "EV Sessions", fontsize=11, fontweight="bold",
        ha="right", va="center")
ax.text(-1, 1.5, "Charging Points", fontsize=11, fontweight="bold",
        ha="right", va="center")
ax.text(-1, 0, "Stations", fontsize=11, fontweight="bold",
        ha="right", va="center")

ax.legend(loc="upper right", fontsize=10)
ax.axis("off")

plt.tight_layout()

# Save figure
os.makedirs("outputs/graphs", exist_ok=True)

output_path = "outputs/graphs/figure_5_3_graph_construction.png"

plt.savefig(output_path, dpi=300, bbox_inches="tight")
print("GRAPH CONSTRUCTION COMPLETED")
print("--------------------------------")
print("EV session nodes      :", len(ev_nodes))
print("Charging point nodes  :", len(charger_nodes))
print("Unique station nodes  :", len(unique_stations))
print("Total graph nodes     :", G.number_of_nodes())
print("Total graph edges     :", G.number_of_edges())
print("Saved at              :", output_path)

plt.show()
