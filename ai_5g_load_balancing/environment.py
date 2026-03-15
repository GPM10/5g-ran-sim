import numpy as np
from utils import estimate_latency_ms, sinr_linear, user_throughput


class NetworkEnvironment:
    def __init__(self, base_stations, users):
        self.base_stations = base_stations
        self.users = users
        self.history = {
            "avg_load": [],
            "avg_throughput": [],
            "max_load": [],
            "avg_latency_ms": [],
            "handover_count": [],
            "avg_sinr_db": [],
        }

    def reset_bs(self):
        for bs in self.base_stations:
            bs.reset()

    def step(self, association_policy):
        for u in self.users:
            u.move()

        self.reset_bs()

        handovers = 0
        for u in self.users:
            previous_bs = u.serving_bs
            selected_bs = association_policy(u, self.base_stations)
            if previous_bs is not None and previous_bs != selected_bs.bs_id:
                handovers += 1
            u.previous_bs = previous_bs
            u.serving_bs = selected_bs.bs_id
            selected_bs.add_user(u)

        throughputs = []
        loads = []
        latencies = []
        sinrs_db = []

        for bs in self.base_stations:
            loads.append(bs.load)
            for u in bs.connected_users:
                sinr_val = sinr_linear(u, bs, self.base_stations)
                sinrs_db.append(10 * np.log10(sinr_val + 1e-9))
                thpt = user_throughput(u, bs, sinr_val)
                throughputs.append(thpt)
                latencies.append(estimate_latency_ms(u, thpt))

        self.history["avg_load"].append(np.mean(loads))
        self.history["max_load"].append(np.max(loads))
        self.history["avg_throughput"].append(
            np.mean(throughputs) if throughputs else 0
        )
        self.history["avg_latency_ms"].append(
            np.mean(latencies) if latencies else 0
        )
        self.history["handover_count"].append(handovers)
        self.history["avg_sinr_db"].append(
            np.mean(sinrs_db) if sinrs_db else -np.inf
        )

    def run(self, association_policy, steps=50):
        self.history = {
            "avg_load": [],
            "avg_throughput": [],
            "max_load": [],
            "avg_latency_ms": [],
            "handover_count": [],
            "avg_sinr_db": [],
        }
        for _ in range(steps):
            self.step(association_policy)
        return self.history
