
import pandas as pd
import numpy as np
from pathlib import Path
from collections import deque

# ==================================================
# 1. CONFIGURATION
# ==================================================

DATA_PATH = Path("outputs/results/processed_acn_data.csv")

OUTPUT_DIR = Path("outputs/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SESSION_OUTPUT = OUTPUT_DIR / "simulation_sessions.csv"
METRICS_OUTPUT = OUTPUT_DIR / "simulation_metrics.csv"

# Experimental assumptions
CHARGER_POWER_KW = 7.2
TIME_STEP_MINUTES = 5

# ==================================================
# 2. CHARGING SIMULATOR
# ==================================================

class ChargingSimulator:

    def __init__(
        self,
        data_path=DATA_PATH,
        charger_power=CHARGER_POWER_KW,
        time_step=TIME_STEP_MINUTES
    ):

        self.data = pd.read_csv(data_path)

        self.charger_power = charger_power
        self.time_step = time_step

        self.data["connectionTime"] = pd.to_datetime(
            self.data["connectionTime"],
            utc=True
        )

        self.data["disconnectTime"] = pd.to_datetime(
            self.data["disconnectTime"],
            utc=True
        )

        self.data["kWhDelivered"] = pd.to_numeric(
            self.data["kWhDelivered"],
            errors="coerce"
        )

        # Keep sessions with valid arrival, departure,
        # and positive historical energy demand.
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

        # Use historical delivered energy as a demand proxy.
        self.data["energy_demand_kwh"] = (
            self.data["kWhDelivered"]
        )

        self.data = self.data.sort_values(
            "connectionTime"
        ).reset_index(drop=True)

        # One simulated charger per unique charging space.
        self.charging_spaces = sorted(
            self.data["spaceID"].astype(str).unique()
        )

        self.num_chargers = len(self.charging_spaces)

        print("\nCHARGING SIMULATOR INITIALIZED")
        print("=" * 50)

        print("Total sessions:", len(self.data))
        print("Charging spaces:", self.num_chargers)
        print("Power per charger:", self.charger_power, "kW")
        print("Time step:", self.time_step, "minutes")

    # ==================================================
    # 3. RUN SIMULATION
    # ==================================================

    def run_fcfs(self):

        print("\nSTARTING FCFS SIMULATION")
        print("=" * 50)

        sessions = self.data.to_dict("records")

        total_sessions = len(sessions)

        # Session tracking
        results = {}

        for i, session in enumerate(sessions):

            results[i] = {
                "sessionID": session["sessionID"],
                "spaceID": session["spaceID"],
                "arrival_time": session["connectionTime"],
                "departure_time": session["disconnectTime"],
                "energy_demand_kwh": session["energy_demand_kwh"],
                "energy_delivered_kwh": 0.0,
                "waiting_time_min": None,
                "charging_time_min": 0.0,
                "assigned_space": None,
                "charging_start_time": None,
                "charging_end_time": None,
                "completed": False
            }

        # Simulation timeline
        start_time = sessions[0]["connectionTime"].floor(
            f"{self.time_step}min"
        )

        end_time = max(
            session["disconnectTime"]
            for session in sessions
        ).ceil(f"{self.time_step}min")

        timeline = pd.date_range(
            start=start_time,
            end=end_time,
            freq=f"{self.time_step}min"
        )

        # Available charging spaces
        free_spaces = deque(self.charging_spaces)

        # Active charging sessions:
        # charger -> session index
        active = {}

        # Waiting EV queue
        waiting_queue = deque()

        arrival_index = 0

        peak_load_kw = 0.0
        total_busy_minutes = 0.0

        # ==================================================
        # 4. TIME-STEP SIMULATION
        # ==================================================

        for current_time in timeline:

            # ----------------------------------------------
            # A. Release completed or departed EVs
            # ----------------------------------------------

            completed_chargers = []

            for charger, idx in list(active.items()):

                session_result = results[idx]

                departure = session_result["departure_time"]

                energy_left = (
                    session_result["energy_demand_kwh"]
                    - session_result["energy_delivered_kwh"]
                )

                if current_time >= departure:
                    session_result["completed"] = (
                        energy_left <= 1e-9
                    )

                    completed_chargers.append(charger)

            for charger in completed_chargers:

                del active[charger]
                free_spaces.append(charger)

            # ----------------------------------------------
            # B. Add newly arrived EVs
            # ----------------------------------------------

            while (
                arrival_index < total_sessions
                and sessions[arrival_index]["connectionTime"]
                <= current_time
            ):

                waiting_queue.append(arrival_index)
                arrival_index += 1

            # ----------------------------------------------
            # C. Remove EVs whose deadlines have passed
            # ----------------------------------------------

            valid_queue = deque()

            while waiting_queue:

                idx = waiting_queue.popleft()

                if results[idx]["departure_time"] <= current_time:
                    continue

                valid_queue.append(idx)

            waiting_queue = valid_queue

            # ----------------------------------------------
            # D. Assign free chargers using FCFS
            # ----------------------------------------------

            while free_spaces and waiting_queue:

                idx = waiting_queue.popleft()

                charger = free_spaces.popleft()

                result = results[idx]

                result["assigned_space"] = charger
                result["charging_start_time"] = current_time

                result["waiting_time_min"] = max(
                    0.0,
                    (
                        current_time - result["arrival_time"]
                    ).total_seconds() / 60
                )

                active[charger] = idx

            # ----------------------------------------------
            # E. Deliver energy to active EVs
            # ----------------------------------------------

            current_load = 0.0

            for charger, idx in list(active.items()):

                result = results[idx]

                remaining_energy = (
                    result["energy_demand_kwh"]
                    - result["energy_delivered_kwh"]
                )

                minutes_available = min(
                    self.time_step,
                    max(
                        0.0,
                        (
                            result["departure_time"]
                            - current_time
                        ).total_seconds() / 60
                    )
                )

                if remaining_energy <= 0 or minutes_available <= 0:
                    continue

                possible_energy = (
                    self.charger_power
                    * minutes_available / 60
                )

                energy_added = min(
                    remaining_energy,
                    possible_energy
                )

                result["energy_delivered_kwh"] += energy_added
                

                charging_minutes = (
                    energy_added / self.charger_power
                ) * 60

                result["charging_time_min"] += charging_minutes

                total_busy_minutes += charging_minutes

                current_load += self.charger_power

            peak_load_kw = max(
                peak_load_kw,
                current_load
            )

        # ==================================================
        # 5. FINALIZE UNASSIGNED AND INCOMPLETE SESSIONS
        # ==================================================

        for idx, result in results.items():

            arrival = result["arrival_time"]
            departure = result["departure_time"]

            if result["waiting_time_min"] is None:

                # EV never received a charging space.
                result["waiting_time_min"] = max(
                    0.0,
                    (departure - arrival).total_seconds() / 60
                )

            result["energy_fulfillment_pct"] = (
                result["energy_delivered_kwh"]
                / result["energy_demand_kwh"]
            ) * 100

            result["energy_shortfall_kwh"] = max(
                0.0,
                result["energy_demand_kwh"]
                - result["energy_delivered_kwh"]
            )

        # ==================================================
        # 6. CALCULATE PERFORMANCE METRICS
        # ==================================================

        results_df = pd.DataFrame(
            list(results.values())
        )

        total_energy_demand = (
            results_df["energy_demand_kwh"].sum()
        )

        total_energy_delivered = (
            results_df["energy_delivered_kwh"].sum()
        )

        average_waiting_time = (
            results_df["waiting_time_min"].mean()
        )

        average_served_waiting_time = (
            results_df.loc[
                results_df["assigned_space"].notna(),
                "waiting_time_min"
            ].mean()
        )

        energy_fulfillment = (
            total_energy_delivered
            / total_energy_demand
        ) * 100 if total_energy_demand > 0 else 0

        completed_sessions = int(
            results_df["completed"].sum()
        )

        session_completion_rate = (
            completed_sessions / total_sessions
        ) * 100

        total_simulation_minutes = (
            end_time - start_time
        ).total_seconds() / 60

        total_available_minutes = (
            self.num_chargers
            * total_simulation_minutes
        )

        charger_utilization = (
            total_busy_minutes
            / total_available_minutes
        ) * 100 if total_available_minutes > 0 else 0

        metrics = {
            "total_sessions": total_sessions,
            "charging_spaces": self.num_chargers,
            "average_waiting_time_min": average_waiting_time,
            "average_served_waiting_time_min": (
                average_served_waiting_time
            ),
            "peak_charging_load_kw": peak_load_kw,
            "energy_fulfillment_pct": energy_fulfillment,
            "completed_sessions": completed_sessions,
            "session_completion_rate_pct": session_completion_rate,
            "charger_utilization_pct": charger_utilization,
            "total_energy_demand_kwh": total_energy_demand,
            "total_energy_delivered_kwh": total_energy_delivered,
            "total_energy_shortfall_kwh": (
                total_energy_demand - total_energy_delivered
            )
        }

        # ==================================================
        # 7. SAVE RESULTS
        # ==================================================

        results_df.to_csv(
            SESSION_OUTPUT,
            index=False
        )

        pd.DataFrame(
            [metrics]
        ).to_csv(
            METRICS_OUTPUT,
            index=False
        )

        # ==================================================
        # 8. DISPLAY RESULTS
        # ==================================================

        print("\nFCFS SIMULATION RESULTS")
        print("=" * 50)

        for name, value in metrics.items():

            if isinstance(value, (float, np.floating)):
                print(f"{name}: {value:.2f}")
            else:
                print(f"{name}: {value}")

        print("\nFILES SAVED")
        print("Session results:", SESSION_OUTPUT)
        print("Metrics:", METRICS_OUTPUT)

        print("\nFCFS SIMULATION COMPLETED!")

        return results_df, metrics


# ==================================================
# 9. MAIN EXECUTION
# ==================================================

if __name__ == "__main__":

    simulator = ChargingSimulator()

    session_results, performance_metrics = (
        simulator.run_fcfs()
    )