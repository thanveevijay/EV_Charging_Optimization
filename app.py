"""
EV CHARGING INTELLIGENCE DASHBOARD
Main method: GCN-DQN | Benchmark: FCFS
Run from the project root with: python app.py
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch

from charging_environment import EVChargingEnvironment
from dqn_agent import DQNAgent
from fcfs_baseline import select_fcfs_action
from gnn_model import POWER_LEVELS

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "outputs" / "results"
GRAPHS = ROOT / "outputs" / "graphs"
CHECKPOINT = ROOT / "models" / "gnn_dqn_model.pth"
DATASET = RESULTS / "processed_acn_data.csv"

SITE_POWER_LIMIT = 30.0
CHARGER_POWER_KW = 7.2
TIME_STEP_MINUTES = 5
NUM_CHARGERS = 54

REQUIRED_SESSION_COLUMNS = [
    "connectionTime", "disconnectTime", "kWhDelivered", "spaceID"
]


def load_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def prepare_sessions(df):
    if df is None or df.empty:
        raise ValueError("Please provide a non-empty session CSV.")
    missing = [c for c in REQUIRED_SESSION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "CSV is missing required columns: " + ", ".join(missing)
        )
    out = df.copy()
    out["connectionTime"] = pd.to_datetime(
        out["connectionTime"], utc=True, errors="coerce"
    )
    out["disconnectTime"] = pd.to_datetime(
        out["disconnectTime"], utc=True, errors="coerce"
    )
    out["kWhDelivered"] = pd.to_numeric(
        out["kWhDelivered"], errors="coerce"
    )
    out = out.dropna(subset=REQUIRED_SESSION_COLUMNS)
    out = out[
        (out["disconnectTime"] > out["connectionTime"])
        & (out["kWhDelivered"] > 0)
    ].copy()
    if out.empty:
        raise ValueError("No valid charging sessions remain after validation.")
    if "sessionID" not in out:
        out["sessionID"] = [f"EV-{i+1:04d}" for i in range(len(out))]
    out["spaceID"] = out["spaceID"].astype(str).str.strip()
    return out.sort_values("connectionTime").reset_index(drop=True)


def load_agent():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Trained GCN-DQN checkpoint not found: {CHECKPOINT}"
        )
    agent = DQNAgent()
    agent.load(str(CHECKPOINT))
    agent.policy_net.eval()
    return agent


def run_scheduler(day_data, algorithm):
    """Run one complete episode using the existing project environment."""
    env = EVChargingEnvironment(
        day_data,
        site_power_limit=SITE_POWER_LIMIT,
        time_step=TIME_STEP_MINUTES,
        charger_power=CHARGER_POWER_KW,
        num_chargers=NUM_CHARGERS
    )
    state = env.reset()
    agent = load_agent() if algorithm == "GCN-DQN" else None

    decisions = 0
    violations = 0

    while state is not None:
        if algorithm == "GCN-DQN":
            action = agent.select_action(state, training=False)
        else:
            action = select_fcfs_action(env, state)

        if action is None:
            break

        state, reward, done, metrics = env.step(action)
        decisions += 1

        if not env.last_interval_constraint_ok:
            violations += 1
        if done:
            break

    metrics = env.get_metrics()
    metrics["algorithm"] = algorithm
    metrics["allocation_decisions"] = decisions
    metrics["constraint_violations"] = violations
    metrics["episode_date"] = str(day_data["connectionTime"].min().date())
    allocations = pd.DataFrame(env.allocation_log)
    return metrics, allocations


def metric_frame(metrics):
    rows = [
        ("Sessions", int(metrics["total_sessions"]), "count"),
        ("Energy demand proxy", round(metrics["energy_demand_kwh"], 2), "kWh"),
        ("Energy delivered", round(metrics["energy_delivered_kwh"], 2), "kWh"),
        ("Energy shortfall", round(metrics["energy_shortfall_kwh"], 2), "kWh"),
        ("Energy fulfillment", round(metrics["energy_fulfillment_pct"], 2), "%"),
        ("Completed sessions", int(metrics["completed_sessions"]), "count"),
        ("Completion rate", round(metrics["completion_rate_pct"], 2), "%"),
        ("Average waiting time", round(metrics["average_waiting_time_min"], 2), "min"),
        ("Peak site load", round(metrics["peak_site_load_kw"], 2), "kW"),
        ("Charger utilization", round(metrics["charger_utilization_pct"], 2), "%"),
        ("Allocation decisions", int(metrics["allocation_decisions"]), "count"),
        ("Constraint violations", int(metrics["constraint_violations"]), "count"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value", "Unit"])


def format_summary(metrics):
    return (
        f"### {metrics['algorithm']} simulation completed\n"
        f"- **{metrics['total_sessions']}** EV sessions processed\n"
        f"- **{metrics['energy_fulfillment_pct']:.2f}%** energy fulfillment\n"
        f"- **{metrics['energy_delivered_kwh']:.2f} kWh** delivered\n"
        f"- **{metrics['average_waiting_time_min']:.2f} min** average waiting time\n"
        f"- **{metrics['peak_site_load_kw']:.2f} kW** peak site load\n"
        f"- **{metrics['constraint_violations']}** power constraint violations"
    )


def run_one(algorithm, uploaded_file, sample_day):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file.name)
    else:
        if not DATASET.exists():
            raise FileNotFoundError(f"Default dataset not found: {DATASET}")
        data = pd.read_csv(DATASET)
        data["connectionTime"] = pd.to_datetime(
            data["connectionTime"], utc=True, errors="coerce"
        )
        data = data.dropna(subset=["connectionTime"])
        data["episode_date"] = data["connectionTime"].dt.strftime("%Y-%m-%d")
        if not sample_day:
            raise ValueError("Choose a test date or upload a session CSV.")
        data = data[data["episode_date"] == sample_day].drop(
            columns=["episode_date"], errors="ignore"
        )
    data = prepare_sessions(data)
    # The environment is designed around one daily episode. Uploaded files
    # containing multiple dates are rejected rather than silently mis-scoped.
    dates = data["connectionTime"].dt.date.nunique()
    if dates != 1:
        raise ValueError(
            "Please upload sessions for one episode date only. "
            "The simulator is evaluated as one daily episode at a time."
        )
    metrics, allocations = run_scheduler(data, algorithm)
    return format_summary(metrics), metric_frame(metrics), allocations


def compare_algorithms(uploaded_file, sample_day):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file.name)
    else:
        if not DATASET.exists():
            raise FileNotFoundError(f"Default dataset not found: {DATASET}")
        data = pd.read_csv(DATASET)
        data["connectionTime"] = pd.to_datetime(
            data["connectionTime"], utc=True, errors="coerce"
        )
        data = data.dropna(subset=["connectionTime"])
        data["episode_date"] = data["connectionTime"].dt.strftime("%Y-%m-%d")
        if not sample_day:
            raise ValueError("Choose a test date or upload a session CSV.")
        data = data[data["episode_date"] == sample_day].drop(
            columns=["episode_date"], errors="ignore"
        )
    data = prepare_sessions(data)
    if data["connectionTime"].dt.date.nunique() != 1:
        raise ValueError("Use one date per comparison so both algorithms receive identical sessions.")

    fcfs, fcfs_log = run_scheduler(data, "FCFS")
    gcn, gcn_log = run_scheduler(data, "GCN-DQN")

    keys = [
        ("Energy fulfillment (%)", "energy_fulfillment_pct", "%"),
        ("Completion rate (%)", "completion_rate_pct", "%"),
        ("Average waiting time (min)", "average_waiting_time_min", "min"),
        ("Peak site load (kW)", "peak_site_load_kw", "kW"),
        ("Charger utilization (%)", "charger_utilization_pct", "%"),
        ("Energy delivered (kWh)", "energy_delivered_kwh", "kWh"),
        ("Energy shortfall (kWh)", "energy_shortfall_kwh", "kWh"),
        ("Constraint violations", "constraint_violations", "count"),
    ]
    table = pd.DataFrame([
        {"Metric": label, "FCFS": round(float(fcfs[key]), 2),
         "GCN-DQN": round(float(gcn[key]), 2), "Unit": unit}
        for label, key, unit in keys
    ])
    chart_df = table[table["Metric"].isin([
        "Energy fulfillment (%)", "Completion rate (%)",
        "Average waiting time (min)", "Charger utilization (%)"
    ])].melt(id_vars=["Metric", "Unit"], value_vars=["FCFS", "GCN-DQN"],
             var_name="Algorithm", value_name="Value")
    fig = px.bar(
        chart_df, x="Metric", y="Value", color="Algorithm",
        barmode="group", text_auto=".2f",
        title="Same-day scheduling comparison"
    )
    fig.update_layout(template="plotly_white", height=420,
                      legend_title_text="", margin=dict(l=20, r=20, t=60, b=90))
    logs = []
    if not fcfs_log.empty:
        fcfs_log.insert(0, "algorithm", "FCFS")
        logs.append(fcfs_log)
    if not gcn_log.empty:
        gcn_log.insert(0, "algorithm", "GCN-DQN")
        logs.append(gcn_log)
    allocation_log = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
    note = (
        f"**Same input:** {len(data)} sessions on {data['connectionTime'].min().date()}  \n"
        f"**GCN-DQN:** {gcn['energy_fulfillment_pct']:.2f}% fulfillment, "
        f"{gcn['average_waiting_time_min']:.2f} min average wait.  \n"
        f"**FCFS:** {fcfs['energy_fulfillment_pct']:.2f}% fulfillment, "
        f"{fcfs['average_waiting_time_min']:.2f} min average wait.  \n\n"
        "This is a fresh simulation on the selected day. Results may differ from "
        "the saved 74-day held-out evaluation."
    )
    return note, table, fig, allocation_log


def overview():
    processed = load_csv(DATASET)
    comparison = load_csv(RESULTS / "algorithm_comparison.csv")
    history = load_csv(RESULTS / "training_history.csv")
    predictions = load_csv(RESULTS / "demand_predictions.csv")

    stats = {
        "Dataset sessions": f"{len(processed):,}" if not processed.empty else "—",
        "Charging spaces": str(processed["spaceID"].nunique()) if "spaceID" in processed else "—",
        "Training episodes": str(len(history)) if not history.empty else "—",
        "Test dates": str(len(load_csv(RESULTS / "gnn_dqn_test_results.csv"))) if (RESULTS / "gnn_dqn_test_results.csv").exists() else "—",
    }
    cards = " | ".join(f"**{k}:** {v}" for k, v in stats.items())

    if not history.empty:
        xcol = "episode" if "episode" in history.columns else history.columns[0]
        reward_col = next((c for c in history.columns if "reward" in c.lower()), None)
        loss_col = next((c for c in history.columns if "loss" in c.lower()), None)
        fig = go.Figure()
        if reward_col:
            fig.add_trace(go.Scatter(x=history[xcol], y=history[reward_col],
                                     mode="lines+markers", name="Episode reward"))
        if loss_col:
            fig.add_trace(go.Scatter(x=history[xcol], y=history[loss_col],
                                     mode="lines", name="Training loss",
                                     yaxis="y2"))
        fig.update_layout(
            title="GCN-DQN training history", template="plotly_white",
            yaxis=dict(title="Reward"), yaxis2=dict(title="Loss", overlaying="y", side="right"),
            height=360
        )
    else:
        fig = go.Figure()
        fig.update_layout(title="Training history file not found", template="plotly_white")

    comp_fig = go.Figure()
    if not comparison.empty:
        # Display the stored comparison table; do not recalculate published results.
        comp_fig = go.Figure()
        for _, row in comparison.iterrows():
            metric = str(row.get("metric", row.get("Metric", "")))
            fcfs_value = row.get("FCFS", row.get("fcfs"))
            gcn_value = row.get("GCN-DQN", row.get("GNN-DQN", row.get("gcn_dqn")))
            if pd.notna(fcfs_value) and pd.notna(gcn_value):
                comp_fig.add_trace(go.Bar(name="FCFS", x=[metric], y=[fcfs_value]))
                comp_fig.add_trace(go.Bar(name="GCN-DQN", x=[metric], y=[gcn_value]))
        comp_fig.update_layout(barmode="group", title="Saved FCFS vs GCN-DQN evaluation",
                               template="plotly_white", height=360)
    else:
        comp_fig.update_layout(title="Saved comparison CSV not found", template="plotly_white")

    pred_fig = go.Figure()
    if not predictions.empty and {"timestamp", "actual_ev_arrivals", "predicted_ev_arrivals"}.issubset(predictions.columns):
        pred_fig.add_trace(go.Scatter(x=predictions["timestamp"], y=predictions["actual_ev_arrivals"], name="Actual EV arrivals"))
        pred_fig.add_trace(go.Scatter(x=predictions["timestamp"], y=predictions["predicted_ev_arrivals"], name="Predicted EV arrivals"))
        pred_fig.update_layout(title="Supporting demand forecast — historical test predictions",
                               template="plotly_white", height=340)
    else:
        pred_fig.update_layout(title="Demand prediction results not available", template="plotly_white")
    return cards, fig, comp_fig, pred_fig


def available_dates():
    if not DATASET.exists():
        return []
    df = pd.read_csv(DATASET, usecols=["connectionTime"])
    ts = pd.to_datetime(df["connectionTime"], utc=True, errors="coerce").dropna()
    return sorted(ts.dt.strftime("%Y-%m-%d").unique().tolist())


CSS = """
.gradio-container {max-width: 1240px !important; margin: auto !important;}
#hero {padding: 28px 30px; border-radius: 18px; background: linear-gradient(115deg,#102b3f,#155e63); color: white;}
#hero h1 {font-size: 32px; margin-bottom: 8px;}
#hero p {font-size: 15px; opacity: .9;}
.panel {border: 1px solid #e5eaf0; border-radius: 14px; padding: 14px;}
"""

with gr.Blocks(title="EV Charging Intelligence", theme=gr.themes.Soft(), css=CSS) as demo:
    gr.HTML("""
    <div id="hero">
      <h1>EV Charging Intelligence</h1>
      <p>Graph-aware, constraint-respecting EV charging optimization powered by GCN-DQN</p>
      <p>MAIN PROPOSED MODEL · FCFS BENCHMARK · ACN DATASET</p>
    </div>
    """)
    gr.Markdown(
        "Explore the trained GCN-DQN scheduler, test charging sessions, and compare it "
        "against FCFS using the same simulation environment."
    )

    with gr.Tab("Overview"):
        overview_stats = gr.Markdown()
        with gr.Row():
            training_plot = gr.Plot()
            saved_comparison_plot = gr.Plot()
        demand_plot = gr.Plot()
        refresh = gr.Button("Refresh project results", variant="secondary")
        refresh.click(overview, outputs=[
            overview_stats, training_plot, saved_comparison_plot, demand_plot
        ])

    with gr.Tab("Run a Scheduler"):
        gr.Markdown("### Test the trained scheduling model\nUpload a one-day CSV or select a date from the processed ACN dataset.")
        with gr.Row():
            algorithm = gr.Radio(["GCN-DQN", "FCFS"], value="GCN-DQN", label="Scheduling algorithm")
            date_choice = gr.Dropdown(choices=available_dates(), label="Test date from ACN dataset")
        session_file = gr.File(label="Optional: upload one-day session CSV", file_types=[".csv"])
        gr.Markdown("Required CSV columns: `connectionTime`, `disconnectTime`, `kWhDelivered`, `spaceID`. Optional: `sessionID`.")
        run_button = gr.Button("Run charging simulation", variant="primary")
        run_summary = gr.Markdown()
        run_metrics = gr.Dataframe(label="Simulation metrics", interactive=False)
        run_allocations = gr.Dataframe(label="Charging allocation log", interactive=False)
        run_button.click(
            run_one, inputs=[algorithm, session_file, date_choice],
            outputs=[run_summary, run_metrics, run_allocations]
        )

    with gr.Tab("Compare Algorithms"):
        gr.Markdown("### GCN-DQN vs FCFS\nBoth algorithms receive the exact same sessions, infrastructure limits and simulation settings.")
        with gr.Row():
            compare_date = gr.Dropdown(choices=available_dates(), label="Test date from ACN dataset")
            compare_file = gr.File(label="Optional: upload one-day session CSV", file_types=[".csv"])
        compare_button = gr.Button("Compare GCN-DQN with FCFS", variant="primary")
        compare_note = gr.Markdown()
        compare_table = gr.Dataframe(label="Side-by-side performance", interactive=False)
        compare_chart = gr.Plot()
        compare_log = gr.Dataframe(label="Combined allocation decisions", interactive=False)
        compare_button.click(
            compare_algorithms, inputs=[compare_file, compare_date],
            outputs=[compare_note, compare_table, compare_chart, compare_log]
        )

    with gr.Tab("Project Notes"):
        gr.Markdown("""
        ### How to interpret the system
        - **GCN-DQN** is the proposed method. The GCN represents the charging site and active EV sessions as a graph; the DQN selects an EV–power allocation action.
        - **FCFS** is the rule-based baseline. It selects the earliest-arriving feasible EV and assigns its highest feasible power.
        - **Demand forecasting** is a separate supporting analysis module. It does not replace the GCN-DQN scheduler.
        - The simulator uses 5-minute intervals, a 30 kW site limit, 7.2 kW maximum charger power and 54 charging spaces by default.

        ### Important experimental limitation
        `kWhDelivered` from ACN is used by the environment as a historical energy-demand proxy. It is not necessarily the driver's original requested energy. The dashboard labels it accordingly.

        ### Evaluation interpretation
        Use the saved 74-day held-out evaluation for the project's reported algorithm comparison. Interactive single-day runs are demonstrations on the selected day and should not be presented as the full test-set result.
        """)

    demo.load(
        overview,
        outputs=[overview_stats, training_plot, saved_comparison_plot, demand_plot]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)
