from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional

from .controller import predictive_mobility_policy
from .environment import NetworkEnvironment
from .sdwan_controller import SdwanController
from .sdwan_edge import SdwanEdge
from .sdwan_wan import WanLink
from .topology import build_network_from_spec, build_reference_network, load_topology_spec


@dataclass
class RegionSpec:
    region_id: str
    topology_path: Optional[str] = None
    has_dpu: bool = False


class MultiRegionSimulator:
    def __init__(self, region_specs: List[RegionSpec]):
        if len(region_specs) < 2:
            raise ValueError("MultiRegionSimulator requires at least two regions")
        self.controller = SdwanController()
        self.region_specs = region_specs
        self._build_regions()
        self._build_links()

    def _build_regions(self):
        for spec in self.region_specs:
            if spec.topology_path:
                spec_data = load_topology_spec(spec.topology_path)
                base_stations, users = build_network_from_spec(spec_data)
            else:
                base_stations, users = build_reference_network()
            env = NetworkEnvironment(base_stations, users)
            edge = SdwanEdge(
                edge_id=f"edge-{spec.region_id}",
                region=spec.region_id,
                has_dpu=spec.has_dpu,
                cpu_capacity_ghz=20.0 if spec.has_dpu else 12.0,
                bandwidth_gbps=25.0 if spec.has_dpu else 15.0,
                encryption_offload=spec.has_dpu,
            )
            self.controller.add_region(spec.region_id, env, edge)

    def _build_links(self):
        specs = self.region_specs
        for i in range(len(specs)):
            for j in range(i + 1, len(specs)):
                src = specs[i]
                dst = specs[j]
                qos = "premium" if (src.has_dpu or dst.has_dpu) else "standard"
                link = WanLink(
                    link_id=f"{src.region_id}-{dst.region_id}",
                    source_region=src.region_id,
                    target_region=dst.region_id,
                    latency_ms=8.0 + 2.0 * j,
                    bandwidth_gbps=15.0,
                    qos_class=qos,
                    jitter_ms=1.5,
                )
                self.controller.add_link(link)

    def step(self, steps: int, association_policy=predictive_mobility_policy):
        for _ in range(steps):
            self.controller.refresh()
            for entry in self.controller.regions.values():
                entry.env.step(association_policy)

    def summary(self):
        result = {}
        for region_id, entry in self.controller.regions.items():
            history = entry.env.history
            result[region_id] = {
                "avg_throughput": history["avg_throughput"][-1] if history["avg_throughput"] else 0.0,
                "avg_latency_ms": history["avg_latency_ms"][-1] if history["avg_latency_ms"] else 0.0,
                "edge_cpu_utilization": entry.edge.cpu_utilization,
                "edge_link_utilization": entry.edge.link_utilization,
            }
        return result


def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-region SD-WAN simulation")
    parser.add_argument("--regions", type=int, default=2, help="Number of regions to create")
    parser.add_argument("--steps", type=int, default=20, help="Simulation steps")
    parser.add_argument("--use-dpu", action="store_true", help="Enable DPUs on alternate regions")
    return parser.parse_args()


def main():
    args = parse_args()
    specs = []
    for idx in range(args.regions):
        specs.append(
            RegionSpec(
                region_id=f"region-{idx}",
                has_dpu=args.use_dpu and idx % 2 == 0,
            )
        )
    simulator = MultiRegionSimulator(specs)
    simulator.step(args.steps)
    summary = simulator.summary()
    for region_id, stats in summary.items():
        print(f"{region_id}: throughput={stats['avg_throughput']:.2f} Mbps "
              f"latency={stats['avg_latency_ms']:.2f} ms "
              f"edge_cpu={stats['edge_cpu_utilization']:.2f} "
              f"edge_link={stats['edge_link_utilization']:.2f}")


if __name__ == "__main__":
    main()
