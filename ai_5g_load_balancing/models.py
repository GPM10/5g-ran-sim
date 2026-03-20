import numpy as np


TRAFFIC_PROFILES = {
    "embb": {
        "demand_mean": 4.0,
        "demand_std": 1.2,
        "latency_budget_ms": 80,
    },
    "urllc": {
        "demand_mean": 2.0,
        "demand_std": 0.4,
        "latency_budget_ms": 10,
    },
    "mmtc": {
        "demand_mean": 0.6,
        "demand_std": 0.2,
        "latency_budget_ms": 200,
    },
}

TIER_CONFIG = {
    "macro": {
        "tx_power_dbm": 46,
        "capacity_mbps": 80,
        "bandwidth_mhz": 40,
        "resource_blocks": 100,
        "path_loss": {
            "los_exp": 2.0,
            "nlos_exp": 3.7,
            "los_intercept": 28.0,
            "nlos_intercept": 36.0,
        },
    },
    "micro": {
        "tx_power_dbm": 33,
        "capacity_mbps": 35,
        "bandwidth_mhz": 20,
        "resource_blocks": 48,
        "path_loss": {
            "los_exp": 2.2,
            "nlos_exp": 3.2,
            "los_intercept": 30.0,
            "nlos_intercept": 38.0,
        },
    },
}


class BaseStation:
    def __init__(
        self,
        bs_id,
        x,
        y,
        tier="macro",
        capacity_mbps=None,
        tx_power_dbm=None,
        bandwidth_mhz=None,
        resource_blocks=None,
    ):
        cfg = TIER_CONFIG.get(tier, TIER_CONFIG["macro"])
        self.bs_id = bs_id
        self.x = x
        self.y = y
        self.tier = tier
        self.capacity_mbps = capacity_mbps if capacity_mbps else cfg["capacity_mbps"]
        self.tx_power_dbm = tx_power_dbm if tx_power_dbm else cfg["tx_power_dbm"]
        self.bandwidth_mhz = bandwidth_mhz if bandwidth_mhz else cfg["bandwidth_mhz"]
        self.resource_blocks = (
            resource_blocks if resource_blocks else cfg["resource_blocks"]
        )
        self.path_loss_params = cfg["path_loss"]
        self.connected_users = []
        self.instant_load = 0.0

    def reset(self):
        self.connected_users = []
        self.instant_load = 0.0

    @property
    def load(self):
        return self.instant_load

    def add_user(self, user):
        self.connected_users.append(user)

    def update_load(self, load_value):
        self.instant_load = load_value


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
        self.backlog_mbits = 0.0
        self.avg_throughput_mbps = 0.1
        self.slot_duration_s = 1.0

    def move(self, area_size=100, step=5, slot_duration_s=1.0):
        dx, dy = np.random.uniform(-step, step, size=2)
        self.x = np.clip(self.x + dx, 0, area_size)
        self.y = np.clip(self.y + dy, 0, area_size)
        self.slot_duration_s = slot_duration_s
        self.generate_traffic()

    def generate_traffic(self):
        arrival = max(self._sample_demand(), 0.1)
        self.demand = arrival
        self.backlog_mbits += arrival * self.slot_duration_s
        return arrival

    def _sample_profile(self):
        return np.random.choice(list(TRAFFIC_PROFILES.keys()))

    def _sample_demand(self):
        cfg = TRAFFIC_PROFILES[self.traffic_profile]
        demand = np.random.normal(cfg["demand_mean"], cfg["demand_std"])
        return float(np.clip(demand, 0.1, None))

    def update_avg_throughput(self, served_mbps, alpha=0.1):
        self.avg_throughput_mbps = (1 - alpha) * self.avg_throughput_mbps + alpha * max(
            served_mbps, 0.0
        )
