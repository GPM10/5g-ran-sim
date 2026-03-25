from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from .models import BaseStation


@dataclass
class RadioUnit:
    ru_id: str
    base_station: BaseStation
    du_id: str
    rf_chains: int = 4
    power_budget_w: float = 1200.0
    temperature_c: float = 35.0
    utilization: float = 0.0

    def update_metrics(self):
        load = getattr(self.base_station, "load", 0.0)
        self.utilization = min(max(load, 0.0), 2.0)
        ambient = 30.0
        self.temperature_c = ambient + 25.0 * self.utilization
        return self.utilization

    def telemetry_labels(self) -> Dict[str, str]:
        return {"ru_id": self.ru_id, "du_id": self.du_id, "bs_id": str(self.base_station.bs_id)}


@dataclass
class DistributedUnit:
    du_id: str
    base_stations: List[BaseStation]
    cu_id: str
    fronthaul_capacity_gbps: float = 10.0
    fronthaul_latency_ms: float = 1.0
    processing_capacity_gbps: float = 5.0
    current_load_gbps: float = 0.0

    def update_load(self, throughput_map: Dict[int, float]) -> float:
        total_mbps = 0.0
        for bs in self.base_stations:
            total_mbps += throughput_map.get(bs.bs_id, 0.0)
        self.current_load_gbps = total_mbps / 1000.0
        return self.current_load_gbps

    @property
    def utilization(self) -> float:
        return min(
            self.current_load_gbps / max(self.processing_capacity_gbps, 1e-6), 2.0
        )

    @property
    def fronthaul_utilization(self) -> float:
        return min(
            self.current_load_gbps / max(self.fronthaul_capacity_gbps, 1e-6), 2.0
        )

    def telemetry_labels(self) -> Dict[str, str]:
        return {"du_id": self.du_id, "cu_id": self.cu_id}


@dataclass
class CentralUnit:
    cu_id: str
    processing_capacity_gbps: float = 20.0
    max_dus: int = 32
    control_plane_load_gbps: float = 0.0
    distributed_units: List[DistributedUnit] = field(default_factory=list)

    def attach_du(self, du: DistributedUnit):
        if du not in self.distributed_units:
            self.distributed_units.append(du)

    def update_utilization(self) -> float:
        aggregate = sum(du.current_load_gbps for du in self.distributed_units)
        self.control_plane_load_gbps = aggregate
        return min(aggregate / max(self.processing_capacity_gbps, 1e-6), 2.0)

    def telemetry_labels(self) -> Dict[str, str]:
        return {
            "cu_id": self.cu_id,
            "du_count": str(len(self.distributed_units)),
        }


def build_cu_du_hierarchy(
    base_stations: Iterable[BaseStation],
    num_cus: int = 2,
    du_defaults: Optional[Dict] = None,
    cu_defaults: Optional[Dict] = None,
) -> Tuple[List[DistributedUnit], List[CentralUnit]]:
    """
    Build a simple CU/DU hierarchy by distributing base stations evenly across CUs.
    """
    du_defaults = du_defaults or {}
    cu_defaults = cu_defaults or {}
    base_stations = list(base_stations)
    if num_cus <= 0:
        num_cus = 1
    central_units: List[CentralUnit] = []
    for idx in range(num_cus):
        cfg = {
            "processing_capacity_gbps": cu_defaults.get("processing_capacity_gbps", 40.0),
            "max_dus": cu_defaults.get("max_dus", 32),
        }
        central_units.append(
            CentralUnit(
                cu_id=f"cu-{idx}",
                processing_capacity_gbps=cfg["processing_capacity_gbps"],
                max_dus=cfg["max_dus"],
            )
        )

    distributed_units: List[DistributedUnit] = []
    for idx, bs in enumerate(base_stations):
        cu = central_units[idx % len(central_units)]
        du = DistributedUnit(
            du_id=f"du-{bs.bs_id}",
            base_stations=[bs],
            cu_id=cu.cu_id,
            fronthaul_capacity_gbps=du_defaults.get("fronthaul_capacity_gbps", 10.0),
            fronthaul_latency_ms=du_defaults.get("fronthaul_latency_ms", 0.8),
            processing_capacity_gbps=du_defaults.get("processing_capacity_gbps", 5.0),
        )
        distributed_units.append(du)
        cu.attach_du(du)

    return distributed_units, central_units


def build_radio_units(distributed_units: Iterable[DistributedUnit]) -> List[RadioUnit]:
    rus: List[RadioUnit] = []
    for du in distributed_units:
        for bs in du.base_stations:
            rus.append(RadioUnit(ru_id=f"ru-{bs.bs_id}", base_station=bs, du_id=du.du_id))
    return rus


__all__ = ["RadioUnit", "DistributedUnit", "CentralUnit", "build_cu_du_hierarchy", "build_radio_units"]
