
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from pathlib import Path

from charging_environment import EVChargingEnvironment
from dqn_agent import DQNAgent


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = Path("outputs/results/processed_acn_data.csv")

MODEL_PATH = Path("models/gnn_dqn_model.pth")

RESULTS_PATH = Path("outputs/results/training_history.csv")

GRAPH_DIR = Path("outputs/graphs")

DEVICE = torch.device("cpu")

SITE_POWER_LIMIT = 30.0
CHARGER_POWER_KW = 7.2
TIME_STEP_MINUTES = 5
NUM_CHARGERS = 54

TRAIN_RATIO = 0.70

MAX_TRAIN_EPISODES = 30

SEED = 42


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_random_seeds(seed=SEED):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def select_training_dates(dates):

    """
    Select a deterministic set of dates from the
    first 70% of chronological dates.
    """

    if len(dates) <= MAX_TRAIN_EPISODES:
        return dates

    indices = np.linspace(
        0,
        len(dates) - 1,
        MAX_TRAIN_EPISODES,
        dtype=int
    )

    return [dates[i] for i in indices]


def safe_metric(metrics, key, default=0.0):

    value = metrics.get(key, default)

    try:

        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except (TypeError, ValueError):

        return default


# ============================================================
# 4. SAVE TRAINING GRAPHS
# ============================================================

def save_training_graphs(history):

    GRAPH_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Graph 1: Training rewards
    # --------------------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.plot(
        history["episode"],
        history["total_reward"],
        marker="o"
    )

    plt.xlabel("Training Episode")
    plt.ylabel("Total Reward")
    plt.title("GNN-DQN Training Rewards")

    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "07_training_rewards.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # Graph 2: Training loss
    # --------------------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.plot(
        history["episode"],
        history["average_loss"],
        marker="o"
    )

    plt.xlabel("Training Episode")
    plt.ylabel("Average DQN Loss")
    plt.title("GNN-DQN Training Loss")

    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "08_training_loss.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # Graph 3: Energy fulfillment
    # --------------------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.plot(
        history["episode"],
        history["energy_fulfillment_pct"],
        marker="o"
    )

    plt.xlabel("Training Episode")
    plt.ylabel("Energy Fulfillment (%)")
    plt.title("GNN-DQN Training Energy Fulfillment")

    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "09_training_fulfillment.png",
        dpi=300
    )

    plt.close()

    print("\nTraining graphs saved successfully.")


# ============================================================
# 5. TRAIN ONE EPISODE
# ============================================================

def train_one_episode(
    agent,
    day_data,
    episode_number,
    global_step
):

    # --------------------------------------------------------
    # Initialize charging environment
    # --------------------------------------------------------

    env = EVChargingEnvironment(
        day_data,
        site_power_limit=SITE_POWER_LIMIT,
        time_step=TIME_STEP_MINUTES,
        charger_power=CHARGER_POWER_KW,
        num_chargers=NUM_CHARGERS
    )

    state = env.reset()

    total_reward = 0.0

    losses = []

    decisions = 0

    # --------------------------------------------------------
    # Interact with environment
    # --------------------------------------------------------

    while state is not None:

        # Select EV and charging power.
        action = agent.select_action(state)

        if action is None:
            break

        # Execute action.
        next_state, reward, done, info = env.step(action)

        transition_done = bool(
            done or next_state is None
        )

        # Store experience.
        agent.remember(
            state,
            action,
            reward,
            next_state,
            transition_done
        )

        total_reward += float(reward)

        decisions += 1
        global_step += 1

        # Optimize GNN-DQN.
        loss = agent.optimize_model()

        if loss is not None:

            if isinstance(loss, torch.Tensor):
                loss = loss.item()

            if np.isfinite(loss):
                losses.append(float(loss))

        # Stop if episode is completed.
        if transition_done:
            break

        state = next_state

    # --------------------------------------------------------
    # Collect episode metrics
    # --------------------------------------------------------

    metrics = env.get_metrics()

    average_loss = (
        float(np.mean(losses))
        if losses
        else 0.0
    )

    episode_result = {

        "episode": episode_number,

        "total_reward": total_reward,

        "average_loss": average_loss,

        "energy_fulfillment_pct": safe_metric(
            metrics,
            "energy_fulfillment_pct"
        ),

        "completion_rate_pct": safe_metric(
            metrics,
            "completion_rate_pct"
        ),

        "average_waiting_time_min": safe_metric(
            metrics,
            "average_waiting_time_min"
        ),

        "peak_site_load_kw": safe_metric(
            metrics,
            "peak_site_load_kw"
        ),

        "total_sessions": safe_metric(
            metrics,
            "total_sessions"
        ),

        "allocation_decisions": decisions,

        "epsilon": safe_metric(
            {"epsilon": agent.epsilon},
            "epsilon"
        )
    }

    return episode_result, global_step


# ============================================================
# 6. MAIN TRAINING FUNCTION
# ============================================================

def main():

    set_random_seeds()

    print("\nGNN-DQN TRAINING")
    print("=" * 65)

    # --------------------------------------------------------
    # Load processed dataset
    # --------------------------------------------------------

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    df["connectionTime"] = pd.to_datetime(
        df["connectionTime"],
        utc=True,
        errors="coerce"
    )

    df["disconnectTime"] = pd.to_datetime(
        df["disconnectTime"],
        utc=True,
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "connectionTime",
            "disconnectTime"
        ]
    ).copy()

    df["episode_date"] = (
        df["connectionTime"].dt.date
    )

    dates = sorted(
        df["episode_date"].dropna().unique()
    )

    if len(dates) < 2:

        raise ValueError(
            "At least two distinct dates are required."
        )

    # --------------------------------------------------------
    # Chronological 70:30 split
    # --------------------------------------------------------

    split_index = int(
        len(dates) * TRAIN_RATIO
    )

    split_index = max(
        1,
        min(split_index, len(dates) - 1)
    )

    training_dates = dates[:split_index]

    testing_dates = dates[split_index:]

    selected_training_dates = select_training_dates(
        training_dates
    )

    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    print("\nDATASET INFORMATION")
    print("-" * 65)

    print("Total sessions:", len(df))
    print("Total dates:", len(dates))

    print(
        "Training dates available:",
        len(training_dates)
    )

    print(
        "Testing dates reserved:",
        len(testing_dates)
    )

    print(
        "Training date range:",
        training_dates[0],
        "to",
        training_dates[-1]
    )

    print(
        "Testing date range:",
        testing_dates[0],
        "to",
        testing_dates[-1]
    )

    print(
        "Training episodes selected:",
        len(selected_training_dates)
    )

    # --------------------------------------------------------
    # Training configuration
    # --------------------------------------------------------

    print("\nTRAINING CONFIGURATION")
    print("-" * 65)

    print("Device:", DEVICE)

    print(
        "Site power limit:",
        SITE_POWER_LIMIT,
        "kW"
    )

    print(
        "Maximum charger power:",
        CHARGER_POWER_KW,
        "kW"
    )

    print(
        "Time step:",
        TIME_STEP_MINUTES,
        "minutes"
    )

    print(
        "Training episodes:",
        len(selected_training_dates)
    )

    print(
        "Random seed:",
        SEED
    )

    print(
        "Target network updates:",
        "Managed by DQNAgent every 100 optimization steps"
    )

    # --------------------------------------------------------
    # Initialize DQN agent
    # --------------------------------------------------------

    agent = DQNAgent()

    history = []

    global_step = 0

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    print("\nSTARTING TRAINING")
    print("=" * 65)

    for episode, day in enumerate(
        selected_training_dates,
        start=1
    ):

        day_data = df[
            df["episode_date"] == day
        ].copy()

        if day_data.empty:
            continue

        print(
            f"\nEpisode {episode}/{len(selected_training_dates)}"
        )

        print("Training date:", day)

        print(
            "Sessions:",
            len(day_data)
        )

        result, global_step = train_one_episode(
            agent,
            day_data,
            episode,
            global_step
        )

        history.append(result)

        # ----------------------------------------------------
        # Episode results
        # ----------------------------------------------------

        print(
            "Total reward:",
            f"{result['total_reward']:.4f}"
        )

        print(
            "Average loss:",
            f"{result['average_loss']:.6f}"
        )

        print(
            "Energy fulfillment:",
            f"{result['energy_fulfillment_pct']:.2f}%"
        )

        print(
            "Completion rate:",
            f"{result['completion_rate_pct']:.2f}%"
        )

        print(
            "Average waiting:",
            f"{result['average_waiting_time_min']:.2f} min"
        )

        print(
            "Allocation decisions:",
            result["allocation_decisions"]
        )

        print(
            "Epsilon:",
            f"{result['epsilon']:.4f}"
        )

    # --------------------------------------------------------
    # Check training completion
    # --------------------------------------------------------

    if not history:

        raise RuntimeError(
            "Training did not produce any episode results."
        )

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        RESULTS_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Save trained model
    # --------------------------------------------------------

    agent.save(str(MODEL_PATH))

    # --------------------------------------------------------
    # Save training graphs
    # --------------------------------------------------------

    save_training_graphs(history_df)

    # --------------------------------------------------------
    # Final training summary
    # --------------------------------------------------------

    print("\n\nTRAINING SUMMARY")
    print("=" * 65)

    print(
        "Completed episodes:",
        len(history)
    )

    print(
        "Total allocation decisions:",
        global_step
    )

    print(
        "DQN optimization steps:",
        agent.steps
    )

    print(
        "Final episode reward:",
        f"{history_df['total_reward'].iloc[-1]:.4f}"
    )

    print(
        "Final episode loss:",
        f"{history_df['average_loss'].iloc[-1]:.6f}"
    )

    print(
        "Final episode energy fulfillment:",
        f"{history_df['energy_fulfillment_pct'].iloc[-1]:.2f}%"
    )

    print(
        "Final episode completion rate:",
        f"{history_df['completion_rate_pct'].iloc[-1]:.2f}%"
    )

    print(
        "Final epsilon:",
        f"{agent.epsilon:.4f}"
    )

    # --------------------------------------------------------
    # Saved files
    # --------------------------------------------------------

    print("\nSAVED FILES")
    print("-" * 65)

    print("Trained model:", MODEL_PATH)

    print("Training history:", RESULTS_PATH)

    print("Training graphs:", GRAPH_DIR)

    print("\nGNN-DQN TRAINING COMPLETED")


# ============================================================
# 7. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()