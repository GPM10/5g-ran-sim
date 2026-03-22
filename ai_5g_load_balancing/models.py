from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


TRAFFIC_PROFILES = {
    "embb": {
        "demand_mean": 4.0,
        "demand_std": 1.2,
        "latency_budget_ms": 80,
        "arrival": "gaussian",
        "burst_shape": 2.0,
    },
    "urllc": {
        "demand_mean": 2.0,
        "demand_std": 0.4,
        "latency_budget_ms": 10,
        "arrival": "deterministic",
        "min_demand": 1.0,
    },
    "mmtc": {
        "demand_mean": 0.6,
        "demand_std": 0.2,
        "latency_budget_ms": 200,
        "arrival": "gamma",
        "burst_shape": 1.2,
    },
    "conversational": {
        "demand_mean": 1.5,
        "demand_std": 0.8,
        "latency_budget_ms": 60,
        "arrival": "pareto",
        "burst_shape": 1.8,
    },
    "massive_iot": {
        "demand_mean": 0.08,
        "demand_std": 0.04,
        "latency_budget_ms": 500,
        "arrival": "spiky",
        "burst_shape": 5.0,
        "min_demand": 0.01,
    },
    "control": {
        "demand_mean": 0.9,
        "demand_std": 0.2,
        "latency_budget_ms": 20,
        "arrival": "deterministic",
        "min_demand": 0.5,
    },
}

MOBILITY_PROFILES = {
    "pedestrian": {"speed_range": (0.5, 1.8), "turn_sigma": 30.0},
    "vehicular": {"speed_range": (5.0, 15.0), "turn_sigma": 6.0},
    "indoor": {"speed_range": (0.2, 0.8), "turn_sigma": 60.0},
    "static": {"speed_range": (0.0, 0.0), "turn_sigma": 0.0},
}


@dataclass
class CarrierConfig:
    name: str
    frequency_ghz: float
    bandwidth_mhz: float
    tx_power_dbm: float
    resource_blocks: int
    capacity_mbps: float
    path_loss_model: Optional[str] = None
    path_loss_params: Optional[Dict] = None

    @classmethod
    def from_dict(cls, data: Dict, defaults: Dict):
        merged = {**defaults, **data}
        if "capacity_mbps" not in merged:
            merged["capacity_mbps"] = merged["bandwidth_mhz"] * 1.5
        return cls(
            name=merged.get("name", "carrier"),
            frequency_ghz=merged.get("frequency_ghz", 3.5),
            bandwidth_mhz=merged.get("bandwidth_mhz", 20.0),
            tx_power_dbm=merged.get("tx_power_dbm", 40.0),
            resource_blocks=merged.get("resource_blocks", 50),
            capacity_mbps=merged["capacity_mbps"],
            path_loss_model=merged.get("path_loss_model"),
            path_loss_params=merged.get("path_loss_params"),
        )


TIER_CONFIG = {
    "macro": {
        "capacity_mbps": 150,
        "path_loss": {
            "model": "cost231",
            "params": {
                "los_exp": 2.0,
                "nlos_exp": 3.7,
                "los_intercept": 28.0,
                "nlos_intercept": 36.0,
            },
        },
        "carriers": [
            {
                "name": "midband",
                "frequency_ghz": 3.5,
                "bandwidth_mhz": 40,
                "tx_power_dbm": 46,
                "resource_blocks": 100,
                "capacity_mbps": 90,
            },
            {
                "name": "mmwave",
                "frequency_ghz": 28.0,
                "bandwidth_mhz": 100,
                "tx_power_dbm": 33,
                "resource_blocks": 120,
                "capacity_mbps": 200,
                "path_loss_model": "nyu_mmwave",
            },
        ],
    },
    "micro": {
        "capacity_mbps": 60,
        "path_loss": {
            "model": "cost231",
            "params": {
                "los_exp": 2.2,
                "nlos_exp": 3.2,
                "los_intercept": 30.0,
                "nlos_intercept": 38.0,
            },
        },
        "carriers": [
            {
                "name": "midband",
                "frequency_ghz": 3.7,
                "bandwidth_mhz": 20,
                "tx_power_dbm": 33,
                "resource_blocks": 48,
                "capacity_mbps": 40,
            }
        ],
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
        z: float = 0.0,
        height_m: float = 25.0,
        azimuth_deg: float = 0.0,
        tilt_deg: float = 5.0,
        beamwidth_deg: float = 65.0,
        path_loss_model: Optional[str] = None,
        path_loss_params: Optional[Dict] = None,
        carriers: Optional[List[Dict]] = None,
    ):
        cfg = TIER_CONFIG.get(tier, TIER_CONFIG["macro"])
        self.bs_id = bs_id
        self.x = x
        self.y = y
        self.z = z
        self.height_m = height_m
        self.azimuth_deg = azimuth_deg
        self.tilt_deg = tilt_deg
        self.beamwidth_deg = beamwidth_deg
        self.tier = tier
        self.path_loss_model = path_loss_model or cfg["path_loss"].get("model", "free_space")
        self.path_loss_params = path_loss_params or cfg["path_loss"].get("params", {})
        carrier_defaults = {
            "path_loss_model": self.path_loss_model,
            "path_loss_params": self.path_loss_params,
        }
        carrier_dicts = carriers if carriers is not None else cfg.get("carriers", [])
        if not carrier_dicts:
            carrier_dicts = [
                {
                    "name": "default",
                    "frequency_ghz": 3.5,
                    "bandwidth_mhz": bandwidth_mhz or 20.0,
                    "tx_power_dbm": tx_power_dbm or 40.0,
                    "resource_blocks": resource_blocks or 50,
                    "capacity_mbps": capacity_mbps or 60.0,
                }
            ]
        self.carriers = [
            CarrierConfig.from_dict(entry, carrier_defaults) for entry in carrier_dicts
        ]
        primary = self.carriers[0]
        self.capacity_mbps = (
            capacity_mbps if capacity_mbps else sum(c.capacity_mbps for c in self.carriers)
        )
        self.tx_power_dbm = tx_power_dbm if tx_power_dbm else primary.tx_power_dbm
        self.bandwidth_mhz = bandwidth_mhz if bandwidth_mhz else primary.bandwidth_mhz
        self.resource_blocks = (
            resource_blocks if resource_blocks else primary.resource_blocks
        )
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

    def get_carrier(self, name: str):
        for carrier in self.carriers:
            if carrier.name == name:
                return carrier
        return None

    def cochannel_carrier(self, target):
        match = self.get_carrier(target.name)
        if match:
            return match
        for carrier in self.carriers:
            if abs(carrier.frequency_ghz - target.frequency_ghz) <= 0.25:
                return carrier
        return None

    @property
    def primary_carrier(self):
        return self.carriers[0]

    def iter_carriers(self):
        return iter(self.carriers)


class UserEquipment:
    def __init__(
        self,
        ue_id,
        x,
        y,
        z: float = 1.5,
        traffic_profile=None,
        environment: str = "urban",
        mobility_profile: str = "pedestrian",
        trajectory: Optional[List[Dict]] = None,
    ):
        self.ue_id = ue_id
        self.x = x
        self.y = y
        self.z = z
        self.height_m = 1.5
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
        self.environment = environment
        self.velocity_m_s = 0.0
        self.heading_deg = np.random.uniform(0, 360)
        self.mobility_profile = mobility_profile or "pedestrian"
        self.trajectory = trajectory or []
        self._trajectory_idx = 0
        self._trajectory_hold = 0

    def move(self, area_size=100, step=5, slot_duration_s=1.0, demand_factor=1.0):
        self.slot_duration_s = slot_duration_s
        if self.trajectory:
            self._follow_trajectory(slot_duration_s, area_size)
        else:
            self._stochastic_walk(area_size, step, slot_duration_s)
        self.generate_traffic(demand_factor=demand_factor)

    def _follow_trajectory(self, slot_duration_s, area_size):
        if not self.trajectory:
            return
        waypoint = self.trajectory[self._trajectory_idx]
        self.x = np.clip(waypoint.get("x", self.x), 0, area_size)
        self.y = np.clip(waypoint.get("y", self.y), 0, area_size)
        speed = waypoint.get("speed_m_s")
        if speed is not None:
            self.velocity_m_s = speed
        hold = waypoint.get("hold_steps", 0)
        if self._trajectory_hold < hold:
            self._trajectory_hold += 1
            return
        self._trajectory_hold = 0
        self._trajectory_idx = (self._trajectory_idx + 1) % len(self.trajectory)

    def _stochastic_walk(self, area_size, step, slot_duration_s):
        mode = MOBILITY_PROFILES.get(self.mobility_profile, MOBILITY_PROFILES["pedestrian"])
        min_speed, max_speed = mode["speed_range"]
        speed = np.random.uniform(min_speed, max_speed)
        self.heading_deg = (self.heading_deg + np.random.normal(0, mode["turn_sigma"])) % 360
        heading_rad = np.deg2rad(self.heading_deg)
        distance = speed * slot_duration_s if max_speed > 0 else 0.0
        dx = distance * np.cos(heading_rad)
        dy = distance * np.sin(heading_rad)
        if max_speed == 0.0:  # static fallback to small jitter if requested
            dx, dy = 0.0, 0.0
        self.x = np.clip(self.x + dx, 0, area_size)
        self.y = np.clip(self.y + dy, 0, area_size)
        self.velocity_m_s = abs(distance) / max(slot_duration_s, 1e-3)

    def generate_traffic(self, demand_factor=1.0):
        cfg = TRAFFIC_PROFILES[self.traffic_profile]
        mean = cfg["demand_mean"] * demand_factor
        arrival = mean
        mode = cfg.get("arrival", "gaussian")
        if mode == "gamma":
            shape = max(cfg.get("burst_shape", 1.5), 0.1)
            scale = max(mean / max(shape, 1e-3), 1e-3)
            arrival = np.random.gamma(shape=shape, scale=scale)
        elif mode == "pareto":
            shape = max(cfg.get("burst_shape", 1.5), 0.1)
            arrival = (np.random.pareto(shape) + 1) * mean / max(shape, 1e-3)
        elif mode == "spiky":
            if np.random.rand() < 0.1:
                arrival = mean * np.random.uniform(5, 12)
            else:
                arrival = np.random.exponential(mean)
        elif mode == "deterministic":
            arrival = mean
        else:
            arrival = np.random.normal(mean, cfg.get("demand_std", 0.5))
        arrival = max(arrival, cfg.get("min_demand", 0.05))
        self.demand = float(arrival)
        self.backlog_mbits += self.demand * self.slot_duration_s
        return self.demand

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
