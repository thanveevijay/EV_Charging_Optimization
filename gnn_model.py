
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv


# ==================================================
# 1. CONFIGURATION
# ==================================================

NODE_FEATURES = 8

HIDDEN_DIM = 32
EMBEDDING_DIM = 32

# Experimental charging-power choices (kW)
POWER_LEVELS = [2.4, 4.8, 7.2]

NUM_POWER_LEVELS = len(POWER_LEVELS)


# ==================================================
# 2. BUILD CHARGING-SITE GRAPH
# ==================================================

def build_charging_graph(
    site_features,
    ev_features,
    action_mask,
    device="cpu"
):
    """
    Constructs a graph for dynamic EV charging.

    Node 0:
        Charging site

    Nodes 1 onwards:
        Active EVs

    Edges:
        Charging site <-> every active EV

    Site features (8):
        1. Site power limit (normalized)
        2. Remaining power budget (normalized)
        3. Current site load (normalized)
        4. Number of active EVs (normalized)
        5. Time of day (normalized)
        6. Day of week (normalized)
        7. Number of waiting EVs (normalized)
        8. Simulation progress (normalized)

    EV features (8):
        1. Remaining energy demand (normalized)
        2. Remaining parking time (normalized)
        3. Urgency (normalized)
        4. Historical energy demand (normalized)
        5. Current allocated power (normalized)
        6. Maximum feasible charging power (normalized)
        7. Historical average power (normalized)
        8. Waiting time (normalized)

    Action mask:
        Shape: (number of EVs, 3)

        True indicates that the corresponding
        EV-power allocation is feasible.
    """

    site_features = torch.as_tensor(
        site_features,
        dtype=torch.float32,
        device=device
    ).reshape(1, NODE_FEATURES)

    ev_features = torch.as_tensor(
        ev_features,
        dtype=torch.float32,
        device=device
    )

    if ev_features.ndim != 2:
        raise ValueError(
            "EV features must be a 2D matrix."
        )

    if ev_features.shape[1] != NODE_FEATURES:
        raise ValueError(
            f"Expected {NODE_FEATURES} EV features, "
            f"received {ev_features.shape[1]}."
        )

    num_evs = ev_features.shape[0]

    if num_evs == 0:
        raise ValueError(
            "At least one active EV is required."
        )

    action_mask = torch.as_tensor(
        action_mask,
        dtype=torch.bool,
        device=device
    )

    if action_mask.shape != (
        num_evs,
        NUM_POWER_LEVELS
    ):
        raise ValueError(
            f"Expected action mask shape "
            f"({num_evs}, {NUM_POWER_LEVELS}), "
            f"received {tuple(action_mask.shape)}."
        )

    # Combine site and EV node features.
    node_features = torch.cat(
        [site_features, ev_features],
        dim=0
    )

    # Create bidirectional site-EV edges.
    site_indices = torch.zeros(
        num_evs,
        dtype=torch.long,
        device=device
    )

    ev_indices = torch.arange(
        1,
        num_evs + 1,
        dtype=torch.long,
        device=device
    )

    forward_edges = torch.stack(
        [site_indices, ev_indices],
        dim=0
    )

    backward_edges = torch.stack(
        [ev_indices, site_indices],
        dim=0
    )

    edge_index = torch.cat(
        [
            forward_edges,
            backward_edges
        ],
        dim=1
    )

    return (
        node_features,
        edge_index,
        action_mask
    )


# ==================================================
# 3. GRAPH NEURAL NETWORK
# ==================================================

class EVChargingGNN(nn.Module):

    def __init__(
        self,
        input_dim=NODE_FEATURES,
        hidden_dim=HIDDEN_DIM,
        embedding_dim=EMBEDDING_DIM
    ):

        super().__init__()

        # Graph convolution layer 1
        self.gcn1 = GCNConv(
            input_dim,
            hidden_dim
        )

        # Graph convolution layer 2
        self.gcn2 = GCNConv(
            hidden_dim,
            embedding_dim
        )

        # EV-specific Q-value prediction.
        #
        # Each EV receives three Q-values:
        # 2.4 kW, 4.8 kW and 7.2 kW.

        self.q_network = nn.Sequential(

            nn.Linear(
                embedding_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                NUM_POWER_LEVELS
            )
        )

    # ==================================================
    # 4. FORWARD PASS
    # ==================================================

    def forward(
        self,
        node_features,
        edge_index,
        action_mask=None
    ):

        # First graph convolution
        x = self.gcn1(
            node_features,
            edge_index
        )

        x = F.relu(x)

        # Second graph convolution
        x = self.gcn2(
            x,
            edge_index
        )

        x = F.relu(x)

        # Exclude node 0 (charging site).
        ev_embeddings = x[1:]

        # Predict Q-values for each EV.
        #
        # Output shape:
        # (number of EVs, 3)

        q_values = self.q_network(
            ev_embeddings
        )

        # Mask infeasible power allocations.
        if action_mask is not None:

            q_values = q_values.masked_fill(
                ~action_mask,
                -1e9
            )

        return q_values, x


# ==================================================
# 5. MODEL TEST
# ==================================================

if __name__ == "__main__":

    print("\nTESTING DYNAMIC EV CHARGING GNN")
    print("=" * 55)

    device = torch.device("cpu")

    model = EVChargingGNN().to(device)

    # ----------------------------------------------
    # Example charging-site features
    # ----------------------------------------------

    site_features = [
        0.75,  # Site power limit
        0.55,  # Remaining power budget
        0.45,  # Current load
        0.20,  # Active EV count
        0.60,  # Time of day
        0.30,  # Day of week
        0.10,  # Waiting EV count
        0.50   # Simulation progress
    ]

    # ----------------------------------------------
    # Four active EVs
    # ----------------------------------------------

    ev_features = [

        [
            0.80,  # Remaining energy
            0.20,  # Remaining parking time
            0.90,  # Urgency
            0.70,  # Historical energy
            0.00,  # Current allocated power
            1.00,  # Maximum feasible power
            0.80,  # Historical average power
            0.30   # Waiting time
        ],

        [
            0.60,
            0.80,
            0.30,
            0.60,
            0.00,
            1.00,
            0.50,
            0.10
        ],

        [
            0.40,
            0.50,
            0.50,
            0.40,
            0.00,
            1.00,
            0.60,
            0.20
        ],

        [
            0.90,
            0.30,
            0.80,
            0.90,
            0.00,
            1.00,
            0.70,
            0.40
        ]
    ]

    # ----------------------------------------------
    # Feasible action mask
    # ----------------------------------------------

    action_mask = [

        [True, True, True],

        [True, True, True],

        [True, True, False],

        [True, False, False]
    ]

    # Build graph
    state = build_charging_graph(
        site_features,
        ev_features,
        action_mask,
        device
    )

    node_features, edge_index, mask = state

    # Forward pass
    q_values, embeddings = model(
        node_features,
        edge_index,
        mask
    )

    print("\nGRAPH INFORMATION")
    print("-" * 55)

    print(
        "Number of graph nodes:",
        node_features.shape[0]
    )

    print(
        "Number of graph edges:",
        edge_index.shape[1]
    )

    print(
        "Node feature shape:",
        node_features.shape
    )

    print(
        "Embedding shape:",
        embeddings.shape
    )

    print("\nMODEL OUTPUT")
    print("-" * 55)

    print(
        "Number of active EVs:",
        q_values.shape[0]
    )

    print(
        "Power allocation options:",
        POWER_LEVELS
    )

    print(
        "Q-value matrix shape:",
        q_values.shape
    )

    print("\nQ-VALUES")

    print(
        q_values.detach().numpy()
    )

    # ----------------------------------------------
    # Verify infeasible actions are masked
    # ----------------------------------------------

    print("\nACTION MASK VALIDATION")

    for ev_index in range(len(ev_features)):

        valid_actions = torch.where(
            mask[ev_index]
        )[0].tolist()

        print(
            f"EV {ev_index + 1}: "
            f"Feasible power options = "
            f"{[POWER_LEVELS[i] for i in valid_actions]}"
        )

    print("\nGNN MODEL TEST COMPLETED SUCCESSFULLY!")
