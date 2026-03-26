from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class WanLink:
    link_id: str
    source_region: str
    target_region: str
    latency_ms: float
    bandwidth_gbps: float
    qos_class: str = "standard"
    jitter_ms: float = 1.0
    packet_loss: float = 0.001
    utilization: float = 0.0
    reserved_bandwidth_gbps: float = 0.0
    _queue_depth: float = 0.0

    def update_utilization(self, throughput_gbps: float, queue_depth: float):
        self.utilization = min(max(throughput_gbps / max(self.bandwidth_gbps, 1e-6), 0.0), 1.5)
        self._queue_depth = max(queue_depth, 0.0)

    def telemetry(self) -> Dict[str, float]:
        return {
            "latency_ms": self.latency_ms,
            "jitter_ms": self.jitter_ms,
            "utilization": self.utilization,
            "queue_depth": self._queue_depth,
            "reserved_gbps": self.reserved_bandwidth_gbps,
        }

    def reserve(self, amount_gbps: float):
        self.reserved_bandwidth_gbps = min(max(amount_gbps, 0.0), self.bandwidth_gbps)
