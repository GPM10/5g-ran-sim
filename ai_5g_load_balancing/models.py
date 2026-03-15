import numpy as np


TRAFFIC_PROFILES = {
    "embb": {"demand_mean": 4.0, "demand_std": 1.2, "latency_budget_ms": 80},
    "urllc": {"demand_mean": 2.0, "demand_std": 0.4, "latency_budget_ms": 10},
    "mmtc": {"demand_mean": 0.6, "demand_std": 0.2, "latency_budget_ms": 200},
}


class BaseStation:
    def __init__(self, bs_id, x, y, capacity_mbps=50, tx_power_dbm=43):
        self.bs_id = bs_id
        self.x = x
        self.y = y
        self.capacity_mbps = capacity_mbps
        self.tx_power_dbm = tx_power_dbm
        self.connected_users = []

    def reset(self):
        self.connected_users = []

    @property
    def load(self):
        total_demand = sum(u.demand for u in self.connected_users)
        return total_demand / self.capacity_mbps

    def add_user(self, user):
        self.connected_users.append(user)


class UserEquipment:
    def __init__(self, ue_id, x, y, traffic_profile=None):
        self.ue_id = ue_id
        self.x = x
        self.y = y
        self.traffic_profile = (
            traffic_profile if traffic_profile else self._sample_profile()
        )
        self.latency_budget_ms = TRAFFIC_PROFILES[self.traffic_profile][
            "latency_budget_ms"
        ]
        self.demand = self._sample_demand()
        self.serving_bs = None
        self.previous_bs = None

    def move(self, area_size=100, step=5):
        dx, dy = np.random.uniform(-step, step, size=2)
        self.x = np.clip(self.x + dx, 0, area_size)
        self.y = np.clip(self.y + dy, 0, area_size)
        self.demand = self._sample_demand()

    def _sample_profile(self):
        return np.random.choice(list(TRAFFIC_PROFILES.keys()))

    def _sample_demand(self):
        cfg = TRAFFIC_PROFILES[self.traffic_profile]
        demand = np.random.normal(cfg["demand_mean"], cfg["demand_std"])
        return float(np.clip(demand, 0.1, None))
