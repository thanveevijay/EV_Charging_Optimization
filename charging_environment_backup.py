import numpy as np
import pandas as pd
import torch

from pathlib import Path

from gnn_model import (
    build_charging_graph,
    POWER_LEVELS,
    NUM_POWER_LEVELS
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = Path("outputs/results/processed_acn_data.csv")

CHARGER_POWER_KW = 7.2
TIME_STEP_MINUTES = 5
NUM_CHARGERS = 54

POWER_LEVELS = np.asarray(POWER_LEVELS, dtype=np.float64)

DEVICE = torch.device("cpu")
ENERGY_TOLERANCE = 1e-6


# ============================================================
# 2. DYNAMIC EV CHARGING ENVIRONMENT
# ============================================================

class EVChargingEnvironment:

    def __init__(
        self,
        data,
        site_power_limit=30.0,
        time_step=TIME_STEP_MINUTES,
        charger_power=CHARGER_POWER_KW,
        num_chargers=NUM_CHARGERS,
        validate_spaces=True
    ):

        self.data = data.copy()

        self.site_power_limit = float(site_power_limit)
        self.time_step = int(time_step)
        self.charger_power = float(charger_power)
        self.num_chargers = int(num_chargers)

        if self.site_power_limit <= 0:
            raise ValueError("Site power limit must be positive.")

        if self.time_step <= 0:
            raise ValueError("Time step must be positive.")

        if self.charger_power <= 0:
            raise ValueError("Charger power must be positive.")

        if self.num_chargers <= 0:
            raise ValueError("Number of chargers must be positive.")

        if np.max(POWER_LEVELS) > self.charger_power:
            raise ValueError("Power level exceeds charger capacity.")

        required_columns = [
            "connectionTime",
            "disconnectTime",
            "kWhDelivered",
            "spaceID"
        ]

        missing = [
            col for col in required_columns
            if col not in self.data.columns
        ]

        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # ----------------------------------------------------
        # Prepare timestamps and numerical columns
        # ----------------------------------------------------

        self.data["connectionTime"] = pd.to_datetime(
            self.data["connectionTime"],
            utc=True,
            errors="coerce"
        )

        self.data["disconnectTime"] = pd.to_datetime(
            self.data["disconnectTime"],
            utc=True,
            errors="coerce"
        )

        self.data["kWhDelivered"] = pd.to_numeric(
            self.data["kWhDelivered"],
            errors="coerce"
        )

        self.data = self.data.dropna(
            subset=[
                "connectionTime",
                "disconnectTime",
                "kWhDelivered",
                "spaceID"
            ]
        ).copy()

        self.data = self.data[
            (self.data["kWhDelivered"] > 0)
            & (
                self.data["disconnectTime"]
                > self.data["connectionTime"]
            )
        ].copy()

        self.data["spaceID"] = (
            self.data["spaceID"].astype(str).str.strip()
        )

        self.data = self.data[
            self.data["spaceID"] != ""
        ].copy()

        if "average_power_kw" in self.data.columns:
            self.data["average_power_kw"] = pd.to_numeric(
                self.data["average_power_kw"],
                errors="coerce"
            )
        else:
            self.data["average_power_kw"] = np.nan

        self.data = self.data.sort_values(
            "connectionTime"
        ).reset_index(drop=True)

        if self.data.empty:
            raise ValueError("No valid charging sessions found.")

        # Historical delivered energy is used as a proxy for
        # energy demand. It is not a confirmed requested amount.
        self.data["energy_demand_kwh"] = (
            self.data["kWhDelivered"].astype(float)
        )

        self.total_sessions = len(self.data)

        unique_spaces = self.data["spaceID"].nunique()

        if unique_spaces > self.num_chargers:
            raise ValueError(
                f"Found {unique_spaces} spaces, but capacity is "
                f"{self.num_chargers}."
            )

        self.space_to_index = {
            space: index
            for index, space in enumerate(
                sorted(self.data["spaceID"].unique())
            )
        }

        self.data["space_index"] = (
            self.data["spaceID"].map(self.space_to_index)
        )

        self.data = self.data.reset_index(drop=True)

        # Validate physical space overlaps.
        self.space_conflicts = self._find_space_conflicts()

        if validate_spaces and self.space_conflicts:
            raise ValueError(
                "Overlapping sessions found on the same physical "
                "charging space. Inspect the source data before "
                "running the simulator.\n"
                + pd.DataFrame(
                    self.space_conflicts[:10]
                ).to_string(index=False)
            )

        print("\nDYNAMIC EV CHARGING ENVIRONMENT")
        print("=" * 60)
        print("Total sessions:", self.total_sessions)
        print("Infrastructure spaces:", self.num_chargers)
        print("Spaces in selected data:", unique_spaces)
        print("Site power limit:", self.site_power_limit, "kW")
        print("Maximum charger power:", self.charger_power, "kW")
        print("Time step:", self.time_step, "minutes")
        print("Physical space conflicts:", len(self.space_conflicts))

    # ========================================================
    # 3. PHYSICAL SPACE VALIDATION
    # ========================================================

    def _find_space_conflicts(self):

        conflicts = []

        for space, group in self.data.groupby("spaceID"):

            group = group.sort_values("connectionTime")

            previous_departure = None
            previous_session = None

            for _, row in group.iterrows():

                if (
                    previous_departure is not None
                    and row["connectionTime"] < previous_departure
                ):

                    conflicts.append({
                        "spaceID": space,
                        "previous_session": previous_session,
                        "next_session": row.get("sessionID", "unknown"),
                        "previous_departure": previous_departure,
                        "next_arrival": row["connectionTime"]
                    })

                if (
                    previous_departure is None
                    or row["disconnectTime"] > previous_departure
                ):
                    previous_departure = row["disconnectTime"]
                    previous_session = row.get("sessionID", "unknown")

        return conflicts

    # ========================================================
    # 4. RESET EPISODE
    # ========================================================

    def reset(self, start_time=None):

        if start_time is None:

            self.current_time = (
                self.data["connectionTime"].min()
                .floor(f"{self.time_step}min")
            )

        else:

            self.current_time = pd.Timestamp(start_time)

            if self.current_time.tzinfo is None:
                self.current_time = self.current_time.tz_localize("UTC")
            else:
                self.current_time = self.current_time.tz_convert("UTC")

        self.episode_start_time = self.current_time

        self.end_time = (
            self.data["disconnectTime"].max()
            .ceil(f"{self.time_step}min")
        )

        if self.current_time >= self.end_time:
            raise ValueError("Invalid episode start time.")

        self.session_arrival = (
            self.data["connectionTime"].tolist()
        )

        self.session_departure = (
            self.data["disconnectTime"].tolist()
        )

        self.session_energy = (
            self.data["energy_demand_kwh"]
            .to_numpy(dtype=np.float64)
        )

        self.session_power = (
            self.data["average_power_kw"]
            .to_numpy(dtype=np.float64)
        )

        self.session_spaces = (
            self.data["spaceID"].tolist()
        )

        self.session_space_indices = (
            self.data["space_index"]
            .to_numpy(dtype=int)
        )

        # Only sessions departing after the episode start
        # are included in episode-level performance metrics.
        self.episode_mask = np.array(
            [
                departure > self.episode_start_time
                for departure in self.session_departure
            ],
            dtype=bool
        )

        self.episode_session_count = int(
            self.episode_mask.sum()
        )

        # For a peak-time experiment, historical energy is
        # treated as a residual-demand proxy from the start.
        self.energy_remaining = self.session_energy.copy()

        self.energy_remaining[~self.episode_mask] = 0.0

        self.energy_delivered = np.zeros(
            self.total_sessions,
            dtype=np.float64
        )

        self.arrived = np.array(
            [
                arrival <= self.current_time
                for arrival in self.session_arrival
            ],
            dtype=bool
        )

        self.first_power_time = [
            None for _ in range(self.total_sessions)
        ]

        self.session_completed = np.zeros(
            self.total_sessions,
            dtype=bool
        )

        self.deadline_evaluated = np.zeros(
            self.total_sessions,
            dtype=bool
        )

        self.total_waiting_minutes = 0.0
        self.total_peak_load = 0.0
        self.deadline_missed_sessions = 0
        self.total_deadline_penalty = 0.0

        self.current_load = 0.0
        self.remaining_budget = self.site_power_limit

        self.active_indices = []
        self.unscheduled_indices = []
        self.interval_allocations = {}

        # NEW: Stores committed GCN-DQN allocation decisions.
        self.allocation_log = []

        self.interval_start = None
        self.interval_end = None
        self.interval_number = 0

        self.finished = False

        self.episode_reward = 0.0
        self.interval_reward = 0.0
        self.interval_decisions = 0

        self.last_interval_load = 0.0
        self.last_interval_budget_used = 0.0
        self.last_interval_energy = 0.0
        self.last_interval_allocations = {}
        self.last_interval_waiting = 0.0
        self.last_interval_waiting_ev_count = 0
        self.last_interval_completed = 0
        self.last_interval_missed = 0
        self.last_interval_peak = 0.0
        self.last_interval_constraint_ok = True

        self._prepare_interval()

        return self._get_state()

    # ========================================================
    # 5. PREPARE NEXT INTERVAL
    # ========================================================

    def _prepare_interval(self):

        while self.current_time < self.end_time:

            self.interval_start = self.current_time

            self.interval_end = min(
                self.current_time + pd.Timedelta(
                    minutes=self.time_step
                ),
                self.end_time
            )

            # Include sessions overlapping any part of this
            # interval. Exact overlap is calculated separately.
            self.active_indices = [

                i for i in range(self.total_sessions)

                if (
                    self.episode_mask[i]
                    and self.session_arrival[i] < self.interval_end
                    and self.session_departure[i] > self.interval_start
                )
            ]

            for i in self.active_indices:

                if self.session_arrival[i] <= self.interval_start:
                    self.arrived[i] = True

            self.unscheduled_indices = [

                i for i in self.active_indices

                if self.energy_remaining[i] > ENERGY_TOLERANCE

            ]

            self.interval_allocations = {}

            self.remaining_budget = self.site_power_limit
            self.current_load = 0.0
            self.interval_reward = 0.0
            self.interval_decisions = 0

            if not self.unscheduled_indices:
                self._commit_interval()
                continue

            if self._build_action_mask().any():
                return

            self._commit_interval()

        self.finished = True

    # ========================================================
    # 6. EXACT CONNECTION OVERLAP
    # ========================================================

    def _get_overlap_minutes(self, index):

        start = max(
            self.interval_start,
            self.session_arrival[index]
        )

        end = min(
            self.interval_end,
            self.session_departure[index]
        )

        return max(
            0.0,
            (end - start).total_seconds() / 60.0
        )

    def _get_charging_start(self, index):

        return max(
            self.interval_start,
            self.session_arrival[index]
        )

    # ========================================================
    # 7. URGENCY CALCULATION
    # ========================================================

    def _calculate_urgency(self, index):

        remaining_energy = self.energy_remaining[index]

        remaining_minutes = max(
            0.0,
            (
                self.session_departure[index]
                - self.current_time
            ).total_seconds() / 60.0
        )

        if remaining_minutes <= 0:
            return 1.0

        required_power = (
            remaining_energy / (remaining_minutes / 60.0)
        )

        return float(
            min(required_power / self.charger_power, 1.0)
        )
        # ========================================================
    # 8. BUILD EV FEATURES
    # ========================================================

    def _build_ev_features(self):

        features = []

        for index in self.unscheduled_indices:

            remaining_energy = self.energy_remaining[index]

            effective_start = max(
                self.current_time,
                self.session_arrival[index]
            )

            remaining_minutes = max(
                0.0,
                (
                    self.session_departure[index]
                    - effective_start
                ).total_seconds() / 60.0
            )

            urgency = self._calculate_urgency(index)

            original_energy = self.session_energy[index]

            historical_power = self.session_power[index]

            if not np.isfinite(historical_power):
                historical_power = 0.0

            waiting_minutes = 0.0

            if self.first_power_time[index] is None:

                waiting_minutes = max(
                    0.0,
                    (
                        effective_start
                        - self.session_arrival[index]
                    ).total_seconds() / 60.0
                )

            overlap_hours = max(
                self._get_overlap_minutes(index) / 60.0,
                1e-9
            )

            max_feasible_power = min(
                self.charger_power,
                remaining_energy / overlap_hours
            )

            features.append([

                min(remaining_energy / 70.0, 1.0),

                min(remaining_minutes / 1440.0, 1.0),

                urgency,

                min(original_energy / 70.0, 1.0),

                0.0,

                max_feasible_power / self.charger_power,

                min(
                    max(historical_power, 0.0)
                    / self.charger_power,
                    1.0
                ),

                min(waiting_minutes / 1440.0, 1.0)

            ])

        return np.asarray(features, dtype=np.float32)

    # ========================================================
    # 9. BUILD SITE FEATURES
    # ========================================================

    def _build_site_features(self):

        active_count = len(self.active_indices)

        waiting_count = sum(

            1 for i in self.active_indices

            if (
                self.first_power_time[i] is None
                and self.energy_remaining[i] > ENERGY_TOLERANCE
            )

        )

        return [

            min(
                self.site_power_limit
                / max(self.num_chargers * self.charger_power, 1e-9),
                1.0
            ),

            min(
                self.remaining_budget / self.site_power_limit,
                1.0
            ),

            min(
                self.current_load / self.site_power_limit,
                1.0
            ),

            min(
                active_count / max(self.num_chargers, 1),
                1.0
            ),

            self.current_time.hour / 23.0,

            self.current_time.day_of_week / 6.0,

            min(
                waiting_count / max(self.num_chargers, 1),
                1.0
            ),

            min(
                self.interval_number / 288.0,
                1.0
            )

        ]

    # ========================================================
    # 10. BUILD ACTION MASK
    # ========================================================

    def _build_action_mask(self):

        num_evs = len(self.unscheduled_indices)

        mask = np.zeros(
            (num_evs, NUM_POWER_LEVELS),
            dtype=bool
        )

        for row, index in enumerate(self.unscheduled_indices):

            remaining_energy = self.energy_remaining[index]

            overlap_minutes = self._get_overlap_minutes(index)

            if (
                overlap_minutes <= 0
                or remaining_energy <= ENERGY_TOLERANCE
            ):
                continue

            for power_index, power in enumerate(POWER_LEVELS):

                within_site_limit = (
                    power <= self.remaining_budget + 1e-9
                )

                within_charger_limit = (
                    power <= self.charger_power + 1e-9
                )

                mask[row, power_index] = (
                    within_site_limit
                    and within_charger_limit
                )

        return mask

    # ========================================================
    # 11. GET GNN GRAPH STATE
    # ========================================================

    def _get_state(self):

        if self.finished or not self.unscheduled_indices:
            return None

        site_features = self._build_site_features()

        ev_features = self._build_ev_features()

        action_mask = self._build_action_mask()

        if not action_mask.any():
            return None

        return build_charging_graph(
            site_features,
            ev_features,
            action_mask,
            DEVICE
        )

    # ========================================================
    # 12. APPLY ACTION
    # ========================================================

    def step(self, action):

        if self.finished:
            raise RuntimeError("Episode has already finished.")

        if action is None:
            raise ValueError("A valid action is required.")

        state = self._get_state()

        if state is None:
            raise RuntimeError("No feasible action is available.")

        _, _, action_mask = state

        ev_row, power_index = action

        if (
            ev_row < 0
            or ev_row >= len(self.unscheduled_indices)
            or power_index < 0
            or power_index >= NUM_POWER_LEVELS
        ):
            raise ValueError("Action index is outside valid range.")

        if not action_mask[ev_row, power_index]:
            raise ValueError("Selected EV-power action is infeasible.")

        session_index = self.unscheduled_indices[ev_row]

        selected_power = float(POWER_LEVELS[power_index])

        # Allocate power for this interval.
        self.interval_allocations[session_index] = selected_power

        self.remaining_budget -= selected_power
        self.current_load += selected_power
        self.interval_decisions += 1

        if (
            self.current_load > self.site_power_limit + 1e-9
            or selected_power > self.charger_power + 1e-9
        ):
            raise RuntimeError("Charging power constraint violated.")

        # Calculate energy based on actual connection overlap.
        overlap_hours = (
            self._get_overlap_minutes(session_index) / 60.0
        )

        possible_energy = selected_power * overlap_hours

        actual_energy = min(
            possible_energy,
            self.energy_remaining[session_index]
        )

        energy_reward = actual_energy / 0.6

        urgency = self._calculate_urgency(session_index)

        urgency_reward = energy_reward * urgency

        peak_penalty = (
            0.10
            * (self.current_load / self.site_power_limit) ** 2
        )

        reward = (
            0.80 * energy_reward
            + 0.20 * urgency_reward
            - peak_penalty
        )

        self.episode_reward += reward
        self.interval_reward += reward

        # One allocation per EV in each interval.
        self.unscheduled_indices.pop(ev_row)

        remaining_mask = self._build_action_mask()

        interval_committed = (
            not self.unscheduled_indices
            or not remaining_mask.any()
        )

        if interval_committed:

            deadline_penalty = self._commit_interval()

            reward += deadline_penalty

            self.episode_reward += deadline_penalty

            self._prepare_interval()

        next_state = self._get_state()

        return (
            next_state,
            float(reward),
            self.finished,
            self.get_metrics()
        )

    # ========================================================
    # 13. COMMIT CURRENT INTERVAL
    # ========================================================

    def _commit_interval(self):

        if self.interval_start is None:
            return 0.0

        interval_budget_used = sum(
            self.interval_allocations.values()
        )

        interval_energy = 0.0

        waiting_minutes = 0.0
        waiting_ev_count = 0

        # ----------------------------------------------------
        # Deliver energy only while the EV is connected.
        # ----------------------------------------------------

        for index, power in self.interval_allocations.items():

            overlap_minutes = self._get_overlap_minutes(index)

            overlap_hours = overlap_minutes / 60.0

            possible_energy = power * overlap_hours

            delivered = min(
                possible_energy,
                self.energy_remaining[index]
            )

            # NEW: Record the actual committed GCN-DQN allocation.
            # spaceID refers to the historical ACN charging space.
            # The scheduler selects EV power, not a new physical charger.

            session_id = self.data.iloc[index].get(
                "sessionID", str(index)
            )

            self.allocation_log.append({
                "interval_number": self.interval_number,
                "sessionID": session_id,
                "historical_spaceID": self.session_spaces[index],
                "interval_start": self.interval_start,
                "interval_end": self.interval_end,
                "allocated_power_kw": float(power),
                "energy_delivered_kwh": float(delivered)
            })

            self.energy_delivered[index] += delivered

            self.energy_remaining[index] = max(
                0.0,
                self.energy_remaining[index] - delivered
            )

            interval_energy += delivered

            if self.first_power_time[index] is None:

                self.first_power_time[index] = (
                    self._get_charging_start(index)
                )

        # ----------------------------------------------------
        # Calculate waiting time for EVs not yet charged.
        # Previously accumulated intervals are counted once.
        # ----------------------------------------------------

        for index in self.active_indices:

            if self.first_power_time[index] is not None:
                continue

            if self.energy_remaining[index] <= ENERGY_TOLERANCE:
                continue

            overlap = self._get_overlap_minutes(index)

            if overlap > 0:

                waiting_minutes += overlap
                waiting_ev_count += 1

        self.total_waiting_minutes += waiting_minutes

        # ----------------------------------------------------
        # Evaluate departures and deadline completion.
        # ----------------------------------------------------

        missed_this_interval = 0
        deadline_penalty = 0.0

        for index in range(self.total_sessions):

            if not self.episode_mask[index]:
                continue

            if self.deadline_evaluated[index]:
                continue

            departs_in_interval = (
                self.session_departure[index] > self.interval_start
                and self.session_departure[index] <= self.interval_end
            )

            if not departs_in_interval:
                continue

            self.deadline_evaluated[index] = True

            if self.energy_remaining[index] <= ENERGY_TOLERANCE:

                self.session_completed[index] = True

            else:

                missed_this_interval += 1

                shortfall = self.energy_remaining[index]

                deadline_penalty -= 0.25 * shortfall

        # Count energy-completed sessions.
        self.session_completed |= (
            self.episode_mask
            & (self.energy_remaining <= ENERGY_TOLERANCE)
        )

        self.deadline_missed_sessions += missed_this_interval

        self.total_deadline_penalty += abs(deadline_penalty)

                # ----------------------------------------------------
        # Update site load and metrics.
        # ----------------------------------------------------

        interval_load = interval_budget_used

        self.total_peak_load = max(
            self.total_peak_load,
            interval_load
        )

        self.last_interval_load = interval_load

        self.last_interval_budget_used = interval_budget_used

        self.last_interval_energy = interval_energy

        self.last_interval_allocations = (
            self.interval_allocations.copy()
        )

        self.last_interval_waiting = waiting_minutes

        self.last_interval_waiting_ev_count = waiting_ev_count

        self.last_interval_completed = int(
            self.session_completed.sum()
        )

        self.last_interval_missed = missed_this_interval

        self.last_interval_peak = self.total_peak_load

        # ----------------------------------------------------
        # Validate power constraints.
        # ----------------------------------------------------

        self.last_interval_constraint_ok = (

            interval_load <= self.site_power_limit + 1e-9

            and all(
                power <= self.charger_power + 1e-9
                for power in self.interval_allocations.values()
            )

        )

        if not self.last_interval_constraint_ok:
            raise RuntimeError(
                "Completed interval violated power constraints."
            )

        # Move simulation forward.
        self.current_time = self.interval_end

        self.interval_number += 1

        return float(deadline_penalty)

    # ========================================================
    # 14. PERFORMANCE METRICS
    # ========================================================

    def get_metrics(self):

        total_demand = float(
            self.session_energy[self.episode_mask].sum()
        )

        total_delivered = float(
            self.energy_delivered[self.episode_mask].sum()
        )

        fulfillment = (
            total_delivered / total_demand * 100.0
            if total_demand > 0
            else 0.0
        )

        shortfall = max(
            0.0,
            total_demand - total_delivered
        )

        completed = int(
            self.session_completed[self.episode_mask].sum()
        )

        completion_rate = (
            completed / max(self.episode_session_count, 1) * 100.0
        )

        simulation_minutes = max(
            1.0,
            (
                self.end_time - self.episode_start_time
            ).total_seconds() / 60.0
        )

        total_charging_minutes = (
            total_delivered / self.charger_power * 60.0
        )

        charger_utilization = (
            total_charging_minutes
            / (self.num_chargers * simulation_minutes)
        ) * 100.0

        average_waiting = (
            self.total_waiting_minutes
            / max(self.episode_session_count, 1)
        )

        return {

            "total_sessions": self.episode_session_count,

            "charging_spaces": self.num_chargers,

            "site_power_limit_kw": self.site_power_limit,

            "peak_site_load_kw": self.total_peak_load,

            "energy_demand_kwh": total_demand,

            "energy_delivered_kwh": total_delivered,

            "energy_shortfall_kwh": shortfall,

            "energy_fulfillment_pct": fulfillment,

            "completed_sessions": completed,

            "completion_rate_pct": completion_rate,

            "deadline_missed_sessions": self.deadline_missed_sessions,

            "charger_utilization_pct": charger_utilization,

            "average_waiting_time_min": average_waiting,

            "total_reward": self.episode_reward

        }


# ============================================================
# 15. FIND BUSIEST HISTORICAL TIMESTAMP
# ============================================================

def find_peak_timestamp(data):

    events = []

    for _, row in data.iterrows():

        events.append((row["connectionTime"], 1))
        events.append((row["disconnectTime"], -1))

    # Process departures before arrivals at equal timestamps.
    events.sort(key=lambda item: (item[0], item[1]))

    active = 0
    peak = 0
    peak_time = None

    for timestamp, change in events:

        active += change

        if active > peak:
            peak = active
            peak_time = timestamp

    return peak_time, peak


# ============================================================
# 16. TEST THE ENVIRONMENT AT HISTORICAL PEAK
# ============================================================

if __name__ == "__main__":

    print("\nTESTING CORRECTED DYNAMIC CHARGING ENVIRONMENT")
    print("=" * 65)

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
        subset=["connectionTime", "disconnectTime"]
    ).copy()

    df = df[
        df["disconnectTime"] > df["connectionTime"]
    ].copy()

    df["episode_date"] = df["connectionTime"].dt.date

    daily_results = []

    for date, day_data in df.groupby("episode_date"):

        peak_time, peak_count = find_peak_timestamp(day_data)

        daily_results.append(
            (date, peak_time, peak_count, day_data)
        )

    selected_date, peak_time, peak_count, test_data = max(
        daily_results,
        key=lambda item: item[2]
    )

    print("\nHISTORICAL PEAK")
    print("-" * 65)
    print("Selected date:", selected_date)
    print("Peak timestamp:", peak_time)
    print("Simultaneous EVs:", peak_count)

    env = EVChargingEnvironment(
        test_data,
        site_power_limit=30.0,
        num_chargers=NUM_CHARGERS
    )

    state = env.reset(start_time=peak_time)

    print("\nINITIAL ENVIRONMENT STATE")
    print("-" * 65)

    if state is None:

        print("No feasible charging action at peak timestamp.")

    else:

        node_features, edge_index, action_mask = state

        print("Graph nodes:", node_features.shape[0])
        print("Graph edges:", edge_index.shape[1])
        print("EV candidates:", action_mask.shape[0])
        print("Action mask:", action_mask.shape)
        print("Available actions:", int(action_mask.sum()))

        # ----------------------------------------------------
        # Allocate only during the first interval.
        # This is a constraint test, not a learned policy.
        # ----------------------------------------------------

        print("\nTESTING MULTI-EV POWER ALLOCATION")
        print("-" * 65)

        initial_interval = env.interval_number
        decisions = 0

        while (
            state is not None
            and env.interval_number == initial_interval
        ):

            _, _, mask = state

            valid = torch.nonzero(
                mask,
                as_tuple=False
            )

            if len(valid) == 0:
                break

            # Select the highest feasible power for the
            # first available EV.
            selected = valid[-1]

            action = (
                int(selected[0]),
                int(selected[1])
            )

            state, reward, done, metrics = env.step(action)

            decisions += 1

            if done:
                break

        print("Allocation decisions:", decisions)

        print(
            "Allocated power:",
            round(env.last_interval_budget_used, 2),
            "kW"
        )

        print(
            "Remaining power budget:",
            round(
                env.site_power_limit
                - env.last_interval_budget_used,
                2
            ),
            "kW"
        )

        print(
            "Peak site load:",
            round(env.total_peak_load, 2),
            "kW"
        )

        print(
            "Energy delivered:",
            round(env.last_interval_energy, 4),
            "kWh"
        )

        print(
            "Number of EVs allocated:",
            len(env.last_interval_allocations)
        )

        assert (
            env.total_peak_load <= env.site_power_limit + 1e-9
        ), "Site power limit violated."

        assert all(
            power <= env.charger_power + 1e-9
            for power in env.last_interval_allocations.values()
        ), "Individual charger limit violated."

        assert env.last_interval_constraint_ok

        print("\nSITE POWER CONSTRAINT: PASSED")
        print("INDIVIDUAL CHARGER CONSTRAINT: PASSED")

    print("\nENVIRONMENT TEST COMPLETED")