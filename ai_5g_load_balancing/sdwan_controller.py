from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .sdwan_edge import SdwanEdge
from .sdwan_wan import WanLink


@dataclass
class RegionEntry:
    region_id: str
    env: "NetworkEnvironment"
    edge: SdwanEdge


class SdwanController:
    def __init__(self):
        self.regions: Dict[str, RegionEntry] = {}
        self.links: List[WanLink] = []

    def add_region(self, region_id: str, env: "NetworkEnvironment", edge: SdwanEdge):
        self.regions[region_id] = RegionEntry(region_id=region_id, env=env, edge=edge)

    def add_link(self, link: WanLink):
        self.links.append(link)

    def _region_links(self, region_id: str) -> List[WanLink]:
        return [
            link
            for link in self.links
            if link.source_region == region_id or link.target_region == region_id
        ]

    def refresh(self):
        region_stats: Dict[str, Dict] = {}
        for region_id, entry in self.regions.items():
            env = entry.env
            history = env.history
            throughput = history["avg_throughput"][-1] if history["avg_throughput"] else 0.0
            latency = history["avg_latency_ms"][-1] if history["avg_latency_ms"] else 0.0
            region_stats[region_id] = {
                "throughput_mbps": throughput,
                "latency_ms": latency,
            }
            edge_cpu_load = min(throughput / 50.0, entry.edge.cpu_capacity_ghz)
            entry.edge.update_metrics(edge_cpu_load, throughput / 1000.0)
            self._emit_edge_metrics(entry)

        for link in self.links:
            src_tp = region_stats.get(link.source_region, {}).get("throughput_mbps", 0.0)
            dst_tp = region_stats.get(link.target_region, {}).get("throughput_mbps", 0.0)
            avg_tp = max(src_tp, dst_tp) / 1000.0
            queue_depth = abs(src_tp - dst_tp) / 200.0
            link.update_utilization(avg_tp, queue_depth)
            self._emit_link_metrics(link)

        for region_id, entry in self.regions.items():
            link_context = {
                link.link_id: link.telemetry() for link in self._region_links(region_id)
            }
            wan_context = {
                "edge": {
                    "cpu_utilization": entry.edge.cpu_utilization,
                    "link_utilization": entry.edge.link_utilization,
                    "has_dpu": entry.edge.has_dpu,
                },
                "links": link_context,
            }
            entry.env.set_wan_context(wan_context)

    def _emit_edge_metrics(self, entry: RegionEntry):
        env = entry.env
        if env.telemetry:
            labels = entry.edge.telemetry_labels()
            env.telemetry.emit(
                "sdwan.edge_cpu_utilization",
                float(entry.edge.cpu_utilization),
                labels=labels,
            )
            env.telemetry.emit(
                "sdwan.edge_link_utilization",
                float(entry.edge.link_utilization),
                labels=labels,
            )

    def _emit_link_metrics(self, link: WanLink):
        for region_id in (link.source_region, link.target_region):
            entry = self.regions.get(region_id)
            if not entry or not entry.env.telemetry:
                continue
            labels = {
                "link_id": link.link_id,
                "region": region_id,
                "qos": link.qos_class,
            }
            entry.env.telemetry.emit(
                "sdwan.wan_link_latency_ms",
                float(link.latency_ms),
                labels=labels,
            )
            entry.env.telemetry.emit(
                "sdwan.wan_link_utilization",
                float(link.utilization),
                labels=labels,
            )
