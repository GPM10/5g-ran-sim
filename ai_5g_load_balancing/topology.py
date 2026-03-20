import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from models import BaseStation, UserEquipment


@dataclass
class BaseStationSpec:
    bs_id: int
    x: float
    y: float
    tier: str = "macro"
    capacity_mbps: Optional[float] = None
    tx_power_dbm: Optional[float] = None
    bandwidth_mhz: Optional[float] = None
    resource_blocks: Optional[int] = None


@dataclass
class UserSeed:
    x: float
    y: float
    traffic_profile: Optional[str] = None


@dataclass
class TopologySpec:
    base_stations: Sequence[BaseStationSpec]
    num_users: int = 40
    area_size: int = 100
    user_positions: Optional[Sequence[UserSeed]] = None


def build_reference_network(num_users: int = 40, area_size: int = 100):
    """Default macro+micro topology shared by demos and the RIC/xApp stack."""
    spec = TopologySpec(
        base_stations=[
            BaseStationSpec(0, 20, 20, tier="macro"),
            BaseStationSpec(1, 80, 20, tier="macro"),
            BaseStationSpec(2, 50, 80, tier="macro"),
            BaseStationSpec(3, 35, 55, tier="micro"),
            BaseStationSpec(4, 70, 60, tier="micro"),
        ],
        num_users=num_users,
        area_size=area_size,
    )
    return build_network_from_spec(spec)


def build_network_from_spec(spec: TopologySpec):
    base_stations = [
        BaseStation(
            s.bs_id,
            s.x,
            s.y,
            tier=s.tier,
            capacity_mbps=s.capacity_mbps,
            tx_power_dbm=s.tx_power_dbm,
            bandwidth_mhz=s.bandwidth_mhz,
            resource_blocks=s.resource_blocks,
        )
        for s in spec.base_stations
    ]

    if spec.user_positions:
        users = [
            UserEquipment(
                idx,
                x=seed.x,
                y=seed.y,
                traffic_profile=seed.traffic_profile,
            )
            for idx, seed in enumerate(spec.user_positions)
        ]
    else:
        area = spec.area_size
        num_users = spec.num_users
        users = [
            UserEquipment(i, x=(i * 7) % area, y=(i * 13) % area)
            for i in range(num_users)
        ]
    return base_stations, users


def load_topology_spec(path: str | Path) -> TopologySpec:
    data = json.loads(Path(path).read_text())
    bs_specs = [
        BaseStationSpec(**bs_kwargs)
        for bs_kwargs in data.get("base_stations", [])
    ]
    user_positions = None
    if "user_positions" in data:
        user_positions = [UserSeed(**seed) for seed in data["user_positions"]]
    return TopologySpec(
        base_stations=bs_specs,
        num_users=data.get("num_users", 40),
        area_size=data.get("area_size", 100),
        user_positions=user_positions,
    )
