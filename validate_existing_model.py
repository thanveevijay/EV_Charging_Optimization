"""Non-destructive smoke test for the existing saved GNN-DQN model.
Run from the EV project root: python validate_existing_model.py
This does NOT train, overwrite, or modify the model checkpoint.
"""
from pathlib import Path
import json
import pandas as pd
import torch

from charging_environment import EVChargingEnvironment
from dqn_agent import DQNAgent

DATA_PATH = Path('outputs/results/processed_acn_data.csv')
MODEL_PATH = Path('models/gnn_dqn_model.pth')
OUTPUT_PATH = Path('outputs/results/existing_model_smoke_test.json')
SITE_POWER_LIMIT = 30.0
TIME_STEP = 5
CHARGER_POWER = 7.2
NUM_CHARGERS = 54


def main():
    print('\nEXISTING GNN-DQN MODEL — NON-DESTRUCTIVE VALIDATION')
    print('=' * 62)

    for path in (DATA_PATH, MODEL_PATH):
        if not path.exists():
            raise FileNotFoundError(f'Required file not found: {path}')
        print(f'[OK] Found {path}')

    df = pd.read_csv(DATA_PATH)
    required = {'connectionTime', 'disconnectTime', 'kWhDelivered', 'spaceID'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Dataset is missing columns: {sorted(missing)}')

    df['connectionTime'] = pd.to_datetime(df['connectionTime'], utc=True, errors='coerce')
    df['disconnectTime'] = pd.to_datetime(df['disconnectTime'], utc=True, errors='coerce')
    df = df.dropna(subset=['connectionTime', 'disconnectTime']).copy()
    df['episode_date'] = df['connectionTime'].dt.date
    dates = sorted(df['episode_date'].unique())
    split = int(len(dates) * 0.70)
    test_dates = dates[split:]
    if not test_dates:
        raise ValueError('No held-out test dates found in the processed dataset.')

    test_date = test_dates[0]
    day = df.loc[df['episode_date'] == test_date].copy()
    print(f'[OK] Dataset rows: {len(df):,}; dates: {len(dates)}')
    print(f'[OK] Using first held-out date for smoke test: {test_date}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Inference device: {device}')
    agent = DQNAgent()
    agent.load(str(MODEL_PATH))
    agent.policy_net.eval()

    env = EVChargingEnvironment(
        data=day,
        site_power_limit=SITE_POWER_LIMIT,
        time_step=TIME_STEP,
        charger_power=CHARGER_POWER,
        num_chargers=NUM_CHARGERS,
    )
    state = env.reset()
    decisions = 0
    max_observed_load = 0.0

    while state is not None:
        action = agent.select_action(state, training=False)
        if action is None:
            raise RuntimeError('Agent returned no action while environment had a state.')
        mask = state[2]
        ev_row, power_index = action
        if ev_row < 0 or ev_row >= mask.shape[0] or power_index < 0 or power_index >= mask.shape[1]:
            raise RuntimeError(f'Agent returned out-of-range action: {action}')
        if not bool(mask[ev_row, power_index].item()):
            raise RuntimeError(f'Agent selected masked/infeasible action: {action}')
        state, _, done, _ = env.step(action)
        decisions += 1
        max_observed_load = max(max_observed_load, float(env.current_load))
        if done or state is None:
            break

    metrics = env.get_metrics()
    violations = int(not env.last_interval_constraint_ok)
    result = {
        'test_date': str(test_date),
        'device': str(device),
        'model_path': str(MODEL_PATH),
        'checkpoint_steps': int(agent.steps),
        'epsilon': float(agent.epsilon),
        'allocation_decisions': decisions,
        'allocation_log_rows': len(env.allocation_log),
        'constraint_violations_detected': violations,
        'peak_site_load_kw': float(metrics['peak_site_load_kw']),
        'site_power_limit_kw': SITE_POWER_LIMIT,
        'charger_power_limit_kw': CHARGER_POWER,
        'metrics': {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in metrics.items()},
        'checkpoint_modified': False,
    }

    if result['peak_site_load_kw'] > SITE_POWER_LIMIT + 1e-8:
        raise RuntimeError('Peak site load exceeded configured site power limit.')
    if violations:
        raise RuntimeError('Environment reported a constraint violation.')

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding='utf-8')

    print('\n[PASS] Existing checkpoint loaded and completed a held-out-day smoke test.')
    print(f"Energy fulfillment : {metrics['energy_fulfillment_pct']:.2f}%")
    print(f"Completion rate    : {metrics['completion_rate_pct']:.2f}%")
    print(f"Average wait       : {metrics['average_waiting_time_min']:.2f} min")
    print(f"Peak site load     : {metrics['peak_site_load_kw']:.2f} kW")
    print(f'Allocation decisions: {decisions}')
    print(f'Result saved to     : {OUTPUT_PATH}')
    print('\nNo training was performed and the model checkpoint was not overwritten.')


if __name__ == '__main__':
    main()
