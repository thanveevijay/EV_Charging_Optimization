
import os
import random
import copy
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gnn_model import (
    EVChargingGNN,
    build_charging_graph,
    NUM_POWER_LEVELS,
    POWER_LEVELS
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

DEVICE = torch.device("cpu")

LEARNING_RATE = 0.0005
GAMMA = 0.95

EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9999

BATCH_SIZE = 32
MEMORY_CAPACITY = 10000

TARGET_UPDATE_FREQUENCY = 100


# ============================================================
# 2. STATE UTILITIES
# ============================================================

def clone_state(state):
    """
    Clone graph tensors to store independent
    state snapshots in replay memory.
    """

    if state is None:
        return None

    node_features, edge_index, action_mask = state

    return (
        node_features.detach().clone(),
        edge_index.detach().clone(),
        action_mask.detach().clone()
    )


# ============================================================
# 3. EXPERIENCE REPLAY BUFFER
# ============================================================

class ReplayBuffer:

    def __init__(self, capacity=MEMORY_CAPACITY):

        self.memory = deque(maxlen=capacity)

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        stored_state = clone_state(state)

        stored_next_state = (
            clone_state(next_state)
            if next_state is not None
            else None
        )

        self.memory.append(
            (
                stored_state,
                tuple(action),
                float(reward),
                stored_next_state,
                bool(done)
            )
        )

    def sample(self, batch_size):

        return random.sample(
            self.memory,
            batch_size
        )

    def __len__(self):

        return len(self.memory)


# ============================================================
# 4. GNN-DQN AGENT
# ============================================================

class DQNAgent:

    def __init__(self):

        # Online GNN network
        self.policy_net = EVChargingGNN().to(DEVICE)

        # Target GNN network
        self.target_net = copy.deepcopy(
            self.policy_net
        ).to(DEVICE)

        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=LEARNING_RATE
        )

        self.loss_function = nn.SmoothL1Loss()

        # Replay memory
        self.memory = ReplayBuffer()

        # Exploration configuration
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.gamma = GAMMA

        # Optimization step counter
        self.steps = 0

        print("\nDYNAMIC GNN-DQN AGENT INITIALIZED")
        print("=" * 55)

        print("Learning rate:", LEARNING_RATE)
        print("Discount factor:", GAMMA)
        print("Initial epsilon:", self.epsilon)
        print("Minimum epsilon:", self.epsilon_min)
        print("Epsilon decay:", EPSILON_DECAY)
        print("Replay capacity:", MEMORY_CAPACITY)
        print("Batch size:", BATCH_SIZE)
        print("Target update frequency:", TARGET_UPDATE_FREQUENCY)
        print("Power levels:", POWER_LEVELS)
        print("Device:", DEVICE)

    # ========================================================
    # 5. VALID ACTION EXTRACTION
    # ========================================================

    def get_valid_actions(self, action_mask):

        valid_positions = torch.nonzero(
            action_mask,
            as_tuple=False
        )

        return [
            (
                int(position[0]),
                int(position[1])
            )
            for position in valid_positions
        ]

    # ========================================================
    # 6. ACTION SELECTION
    # ========================================================

    def select_action(
        self,
        state,
        training=True
    ):

        if state is None:
            return None

        node_features, edge_index, action_mask = state

        valid_actions = self.get_valid_actions(
            action_mask
        )

        if not valid_actions:
            return None

        # ----------------------------------------------------
        # Exploration
        # ----------------------------------------------------

        if training and random.random() < self.epsilon:

            return random.choice(valid_actions)

        # ----------------------------------------------------
        # Exploitation
        # ----------------------------------------------------

        self.policy_net.eval()

        with torch.no_grad():

            q_values, _ = self.policy_net(
                node_features,
                edge_index,
                action_mask
            )

            flat_q_values = q_values.reshape(-1)

            selected_index = torch.argmax(
                flat_q_values
            ).item()

            ev_index = (
                selected_index // NUM_POWER_LEVELS
            )

            power_index = (
                selected_index % NUM_POWER_LEVELS
            )

            action = (
                int(ev_index),
                int(power_index)
            )

        return action

    # ========================================================
    # 7. STORE TRANSITION
    # ========================================================

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.memory.push(
            state,
            action,
            reward,
            next_state,
            done
        )

    # ========================================================
    # 8. OPTIMIZATION STEP - DOUBLE DQN
    # ========================================================

    def optimize_model(self):

        if len(self.memory) < BATCH_SIZE:
            return None

        batch = self.memory.sample(BATCH_SIZE)

        self.policy_net.train()

        current_q_values = []
        target_q_values = []

        for (
            state,
            action,
            reward,
            next_state,
            done
        ) in batch:

            node_features, edge_index, action_mask = state

            ev_index, power_index = action

            # ------------------------------------------------
            # Current Q-value
            # ------------------------------------------------

            q_values, _ = self.policy_net(
                node_features,
                edge_index,
                action_mask
            )

            current_q = q_values[
                ev_index,
                power_index
            ]

            target = torch.tensor(
                reward,
                dtype=torch.float32,
                device=DEVICE
            )

            # ------------------------------------------------
            # Double DQN Bellman target
            # ------------------------------------------------

            if not done and next_state is not None:

                (
                    next_features,
                    next_edges,
                    next_mask
                ) = next_state

                if next_mask.any():

                    with torch.no_grad():

                        # Online network selects next action.
                        online_next_q, _ = self.policy_net(
                            next_features,
                            next_edges,
                            next_mask
                        )

                        selected_flat_index = torch.argmax(
                            online_next_q.reshape(-1)
                        )

                        next_ev = int(
                            selected_flat_index.item()
                            // NUM_POWER_LEVELS
                        )

                        next_power = int(
                            selected_flat_index.item()
                            % NUM_POWER_LEVELS
                        )

                        # Target network evaluates that action.
                        target_next_q, _ = self.target_net(
                            next_features,
                            next_edges,
                            next_mask
                        )

                        max_next_q = target_next_q[
                            next_ev,
                            next_power
                        ]

                        target = (
                            target
                            + self.gamma * max_next_q
                        )

            current_q_values.append(current_q)

            target_q_values.append(target)

        # ----------------------------------------------------
        # Calculate loss
        # ----------------------------------------------------

        current_q_values = torch.stack(
            current_q_values
        )

        target_q_values = torch.stack(
            target_q_values
        )

        loss = self.loss_function(
            current_q_values,
            target_q_values
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            max_norm=1.0
        )

        self.optimizer.step()

        self.steps += 1

        # ----------------------------------------------------
        # Target network update
        # ----------------------------------------------------

        if self.steps % TARGET_UPDATE_FREQUENCY == 0:

            self.update_target_network()

        # ----------------------------------------------------
        # Gradual epsilon decay
        # ----------------------------------------------------

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * EPSILON_DECAY
        )

        return float(loss.item())

    # ========================================================
    # 9. UPDATE TARGET NETWORK
    # ========================================================

    def update_target_network(self):

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

        self.target_net.eval()

    # ========================================================
    # 10. SAVE MODEL
    # ========================================================

    def save(self, path="models/gnn_dqn_model.pth"):

        directory = os.path.dirname(path)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True
            )

        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps": self.steps
            },
            path
        )

        print("Model saved:", path)

    # ========================================================
    # 11. LOAD MODEL
    # ========================================================

    def load(self, path="models/gnn_dqn_model.pth"):

        checkpoint = torch.load(
            path,
            map_location=DEVICE,
            weights_only=False
        )

        self.policy_net.load_state_dict(
            checkpoint["policy_net"]
        )

        self.target_net.load_state_dict(
            checkpoint["target_net"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        self.epsilon = checkpoint["epsilon"]
        self.steps = checkpoint["steps"]

        self.target_net.eval()

        print("Model loaded:", path)
        print("Restored epsilon:", self.epsilon)
        print("Restored steps:", self.steps)


# ============================================================
# 12. STRUCTURAL TEST
# ============================================================

if __name__ == "__main__":

    print("\nTESTING DYNAMIC GNN-DQN AGENT")
    print("=" * 55)

    agent = DQNAgent()

    site_features = [
        0.75, 0.55, 0.45, 0.20,
        0.60, 0.30, 0.10, 0.50
    ]

    ev_features = [

        [0.80, 0.20, 0.90, 0.70, 0.00, 1.00, 0.80, 0.30],

        [0.60, 0.80, 0.30, 0.60, 0.00, 1.00, 0.50, 0.10],

        [0.40, 0.50, 0.50, 0.40, 0.00, 1.00, 0.60, 0.20],

        [0.90, 0.30, 0.80, 0.90, 0.00, 1.00, 0.70, 0.40]
    ]

    action_mask = [

        [True, True, True],

        [True, True, True],

        [True, True, False],

        [True, False, False]
    ]

    state = build_charging_graph(
        site_features,
        ev_features,
        action_mask,
        DEVICE
    )

    # Select action
    action = agent.select_action(
        state,
        training=False
    )

    print("\nACTION SELECTION")
    print("-" * 55)

    print("Selected action:", action)

    if action is not None:

        ev_index, power_index = action

        print("Selected EV:", ev_index + 1)

        print(
            "Selected power:",
            POWER_LEVELS[power_index],
            "kW"
        )

    # Store sample transition
    agent.remember(
        state,
        action,
        0.5,
        None,
        True
    )

    print("\nREPLAY BUFFER")
    print("-" * 55)

    print(
        "Stored transitions:",
        len(agent.memory)
    )

    print("\nMODEL INFORMATION")

    parameter_count = sum(
        p.numel()
        for p in agent.policy_net.parameters()
    )

    print(
        "Trainable parameters:",
        parameter_count
    )

    print("\nDYNAMIC GNN-DQN TEST COMPLETED!")
