from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SdwanEdge:
    edge_id: str
    region: str
    has_dpu: bool = False
    cpu_capacity_ghz: float = 16.0
    bandwidth_gbps: float = 20.0
    encryption_offload: bool = False
    _cpu_load_ghz: float = 0.0
    _throughput_gbps: float = 0.0

    def update_metrics(self, cpu_load_ghz: float, throughput_gbps: float):
        self._cpu_load_ghz = min(max(cpu_load_ghz, 0.0), self.cpu_capacity_ghz)
        self._throughput_gbps = min(max(throughput_gbps, 0.0), self.bandwidth_gbps)

    @property
    def cpu_utilization(self) -> float:
        return self._cpu_load_ghz / max(self.cpu_capacity_ghz, 1e-6)

    @property
    def link_utilization(self) -> float:
        return self._throughput_gbps / max(self.bandwidth_gbps, 1e-6)

    def telemetry_labels(self) -> Dict[str, str]:
        return {
            "edge_id": self.edge_id,
            "region": self.region,
            "acceleration": "dpu" if self.has_dpu else "cpu",
        }
