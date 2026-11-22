
from dqn_agent import (
    LEARNING_RATE,
    GAMMA,
    EPSILON_START,
    EPSILON_MIN,
    EPSILON_DECAY,
    BATCH_SIZE,
    TARGET_UPDATE_FREQUENCY
)

from train_agent import MAX_TRAIN_EPISODES, SITE_POWER_LIMIT

print("\n" + "=" * 55)
print("GCN-DQN SIMULATION CONFIGURATION")
print("=" * 55)

print("\n[1] MODEL CONFIGURATION")
print("Graph encoder          : Graph Convolutional Network")
print("DRL algorithm          : Deep Q-Network")

print("\n[2] DQN HYPERPARAMETERS")
print("Learning rate          :", LEARNING_RATE)
print("Discount factor        :", GAMMA)
print("Initial epsilon        :", EPSILON_START)
print("Minimum epsilon        :", EPSILON_MIN)
print("Epsilon decay          :", EPSILON_DECAY)
print("Batch size             :", BATCH_SIZE)
print("Target update frequency:", TARGET_UPDATE_FREQUENCY)

print("\n[3] TRAINING CONFIGURATION")
print("Training episodes      :", MAX_TRAIN_EPISODES)

print("\n[4] SIMULATION CONFIGURATION")
print("Site power limit       :", SITE_POWER_LIMIT, "kW")
print("Scheduling strategies  : FCFS, Earliest Departure,")
print("                         Highest Remaining Demand")

print("\n" + "=" * 55)
print("CONFIGURATION LOADED SUCCESSFULLY")
print("=" * 55)
