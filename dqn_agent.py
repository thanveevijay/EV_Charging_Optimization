import os
import random
import copy
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from gnn_model import (
    EVChargingGNN,
    build_charging_graph,
    NUM_POWER_LEVELS,
    POWER_LEVELS,
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

# Automatically use an available CUDA GPU; otherwise use CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    """Store independent graph snapshots on CPU in replay memory."""
    if state is None:
        return None

    node_features, edge_index, action_mask = state
    return (
        node_features.detach().to("cpu").clone(),
        edge_index.detach().to("cpu").clone(),
        action_mask.detach().to("cpu").clone(),
    )


def state_to_device(state, device):
    """Move a stored graph state to the agent's compute device."""
    if state is None:
        return None

    node_features, edge_index, action_mask = state
    return (
        node_features.to(device),
        edge_index.to(device),
        action_mask.to(device=device, dtype=torch.bool),
    )


# ============================================================
# 3. EXPERIENCE REPLAY BUFFER
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity=MEMORY_CAPACITY):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append(
            (
                clone_state(state),
                tuple(action),
                float(reward),
                clone_state(next_state),
                bool(done),
            )
        )

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# ============================================================
# 4. GNN-DQN AGENT
# ============================================================

class DQNAgent:
    def __init__(self, device=None):
        self.device = torch.device(device) if device is not None else DEVICE

        self.policy_net = EVChargingGNN().to(self.device)
        self.target_net = copy.deepcopy(self.policy_net).to(self.device)
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=LEARNING_RATE,
        )
        self.loss_function = nn.SmoothL1Loss()
        self.memory = ReplayBuffer()

        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.gamma = GAMMA
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
        print("Device:", self.device)

    # ========================================================
    # 5. VALID ACTION EXTRACTION
    # ========================================================

    def get_valid_actions(self, action_mask):
        valid_positions = torch.nonzero(
            action_mask.detach().to("cpu", dtype=torch.bool),
            as_tuple=False,
        )
        return [
            (int(position[0]), int(position[1]))
            for position in valid_positions
        ]

    # ========================================================
    # 6. ACTION SELECTION
    # ========================================================

    def select_action(self, state, training=True):
        if state is None:
            return None

        _, _, action_mask = state
        valid_actions = self.get_valid_actions(action_mask)

        if not valid_actions:
            return None

        # Epsilon-greedy exploration.
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)

        # Greedy action selection over valid actions only.
        inference_state = state_to_device(state, self.device)
        node_features, edge_index, device_mask = inference_state

        was_training = self.policy_net.training
        self.policy_net.eval()

        try:
            with torch.inference_mode():
                q_values, _ = self.policy_net(
                    node_features,
                    edge_index,
                    device_mask,
                )

                q_values = q_values.reshape(-1)
                flat_mask = device_mask.reshape(-1)

                # Defensive guard: never select a masked/invalid action.
                q_values = q_values.masked_fill(~flat_mask, -torch.inf)
                selected_index = int(torch.argmax(q_values).item())

            return (
                selected_index // NUM_POWER_LEVELS,
                selected_index % NUM_POWER_LEVELS,
            )
        finally:
            if was_training:
                self.policy_net.train()

    # ========================================================
    # 7. STORE TRANSITION
    # ========================================================

    def remember(self, state, action, reward, next_state, done):
        if action is None:
            return
        self.memory.push(state, action, reward, next_state, done)

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

        for state, action, reward, next_state, done in batch:
            if state is None:
                continue

            node_features, edge_index, action_mask = state_to_device(
                state, self.device
            )
            ev_index, power_index = action

            q_values, _ = self.policy_net(
                node_features,
                edge_index,
                action_mask,
            )
            current_q = q_values[ev_index, power_index]

            target = torch.tensor(
                reward,
                dtype=torch.float32,
                device=self.device,
            )

            # Double DQN: online network selects; target network evaluates.
            if not done and next_state is not None:
                next_features, next_edges, next_mask = state_to_device(
                    next_state, self.device
                )

                if bool(next_mask.any().item()):
                    with torch.no_grad():
                        online_next_q, _ = self.policy_net(
                            next_features,
                            next_edges,
                            next_mask,
                        )
                        online_next_q = online_next_q.reshape(-1)
                        online_next_q = online_next_q.masked_fill(
                            ~next_mask.reshape(-1), -torch.inf
                        )
                        selected_flat_index = int(
                            torch.argmax(online_next_q).item()
                        )

                        next_ev = selected_flat_index // NUM_POWER_LEVELS
                        next_power = selected_flat_index % NUM_POWER_LEVELS

                        target_next_q, _ = self.target_net(
                            next_features,
                            next_edges,
                            next_mask,
                        )
                        max_next_q = target_next_q[next_ev, next_power]
                        target = target + self.gamma * max_next_q

            current_q_values.append(current_q)
            target_q_values.append(target)

        if not current_q_values:
            return None

        current_q_values = torch.stack(current_q_values)
        target_q_values = torch.stack(target_q_values)

        loss = self.loss_function(current_q_values, target_q_values)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            max_norm=1.0,
        )
        self.optimizer.step()

        self.steps += 1

        if self.steps % TARGET_UPDATE_FREQUENCY == 0:
            self.update_target_network()

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * EPSILON_DECAY,
        )

        return float(loss.detach().cpu().item())

    # ========================================================
    # 9. UPDATE TARGET NETWORK
    # ========================================================

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    # ========================================================
    # 10. SAVE MODEL
    # ========================================================

    def save(self, path="models/gnn_dqn_model.pth"):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        # Keep the same checkpoint keys for compatibility.
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps": self.steps,
            },
            path,
        )
        print("Model saved:", path)

    # ========================================================
    # 11. LOAD MODEL
    # ========================================================

    def load(self, path="models/gnn_dqn_model.pth"):
        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )

        required_keys = {
            "policy_net", "target_net", "optimizer", "epsilon", "steps"
        }
        missing = required_keys.difference(checkpoint.keys())
        if missing:
            raise KeyError(
                "Checkpoint is missing required keys: "
                + ", ".join(sorted(missing))
            )

        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        # Ensure optimizer tensors follow the selected device.
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)

        self.epsilon = float(checkpoint["epsilon"])
        self.steps = int(checkpoint["steps"])
        self.policy_net.eval()
        self.target_net.eval()

        print("Model loaded:", path)
        print("Restored epsilon:", self.epsilon)
        print("Restored steps:", self.steps)
        print("Loaded device:", self.device)


# ============================================================
# 12. STRUCTURAL TEST
# ============================================================

if __name__ == "__main__":
    print("\nTESTING DYNAMIC GNN-DQN AGENT")
    print("=" * 55)

    agent = DQNAgent()

    site_features = [
        0.75, 0.55, 0.45, 0.20,
        0.60, 0.30, 0.10, 0.50,
    ]

    ev_features = [
        [0.80, 0.20, 0.90, 0.70, 0.00, 1.00, 0.80, 0.30],
        [0.60, 0.80, 0.30, 0.60, 0.00, 1.00, 0.50, 0.10],
        [0.40, 0.50, 0.50, 0.40, 0.00, 1.00, 0.60, 0.20],
        [0.90, 0.30, 0.80, 0.90, 0.00, 1.00, 0.70, 0.40],
    ]

    action_mask = [
        [True, True, True],
        [True, True, True],
        [True, True, False],
        [True, False, False],
    ]

    state = build_charging_graph(
        site_features,
        ev_features,
        action_mask,
        agent.device,
    )

    action = agent.select_action(state, training=False)

    print("\nACTION SELECTION")
    print("-" * 55)
    print("Selected action:", action)

    if action is not None:
        ev_index, power_index = action
        print("Selected EV:", ev_index + 1)
        print("Selected power:", POWER_LEVELS[power_index], "kW")

    agent.remember(state, action, 0.5, None, True)

    print("\nREPLAY BUFFER")
    print("-" * 55)
    print("Stored transitions:", len(agent.memory))

    parameter_count = sum(
        parameter.numel()
        for parameter in agent.policy_net.parameters()
    )
    print("\nTrainable parameters:", parameter_count)
    print("\nDYNAMIC GNN-DQN TEST COMPLETED!")
