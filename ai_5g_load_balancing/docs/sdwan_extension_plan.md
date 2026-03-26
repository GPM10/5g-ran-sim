# SD-WAN Extension Plan

This note outlines how to evolve the current RU/DU/CU + interface-aware simulator into an SD-WAN-oriented research platform aligned with the NEWTON MSCA objectives.

## Goals

1. **Interconnected SD-WAN Controllers** – run multiple regional RAN environments that exchange telemetry via SD-WAN links, with a higher-level orchestrator selecting WAN paths subject to QoS classes.
2. **Tenant Intent & Zero-Touch Automation** – expose an intent API where tenants declare latency/reliability/bandwidth requirements; propagate intents into association policies, WAN routing, and compute placement using ML-assisted logic.
3. **Edge Nodes with Acceleration Awareness** – model SD-WAN edge nodes (software- or DPU-accelerated) so the controller can offload flow steering/encryption and reason about compute fabrics.

## Architecture Overview

```
 Tenant Intent API  --->  Intent Processor  --->  SD-WAN Orchestrator
                                  |                    |
                                  v                    v
                       Regional Controller      WAN Fabric (links + edges)
                               |                        |
                         RAN Environment <-----> SD-WAN Edge (DPU/CPU)
```

### Components

| Component | Role |
| --- | --- |
| `SdwanEdge` (new module) | Represents a regional ingress/egress node with compute capacity, DPU acceleration flag, per-tenant slices, and telemetry hooks. |
| `WanLink` (new module) | Models inter-region links with latency/bandwidth/QoS tiers; supports path recomputation and failure injection. |
| `IntentManager` (RIC API) | REST interface where tenants declare intents (`slice_id`, `latency_target_ms`, `bandwidth_mbps`, `priority`). Stores active intents, maps them to policies. |
| `SdwanController` | Global orchestrator deciding WAN paths, selecting edges, and updating per-region policy context (e.g., weight boosts for intents). |
| `MultiRegionSimulator` | Wrapper that instantiates N `NetworkEnvironment` objects plus their edges/links; steps them in lockstep while exchanging WAN traffic summaries. |

## Interfaces & Data Flow

1. **Intent submission** (`POST /intents`):
   - Validate tenant SLA, persist in `IntentManager`.
   - Trigger SD-WAN controller to recompute resource allocations (per-slice weights, WAN path reservations, edge selection).

2. **Regional policy context**:
   - Extend `NetworkEnvironment.policy_context` with:
     - `wan`: dict of current WAN path latency/queue occupancy for the region’s edges.
     - `intents`: currently active intents relevant to the region (slice→desired KPIs).
     - `edge`: acceleration/compute status for its ingress/egress nodes.

3. **Telemetry**:
   - New metrics: `sdwan.wan_link_latency_ms`, `sdwan.wan_link_utilization`, `sdwan.edge_cpu_utilization`, `sdwan.intent_count`.
   - WAN controller exports events when path reconfigurations occur (Prometheus counter + event log).

4. **Tenant-to-policy mapping**:
   - Policies read `context["intents"]` to boost priority for UEs whose slice is violating SLA, or to reroute traffic via low-latency WAN paths.
   - ML loop (bandit/RL) consumes combined telemetry (E2, F1, N2, WAN) to enact “zero-touch” adjustments (e.g., auto-scaling edges, adjusting weights).

## Implementation Plan

1. **Scaffolding**
   - Add `ai_5g_load_balancing/sdwan_edge.py` and `sdwan_wan.py` (edges/links data classes + telemetry emission).
   - Create `services/intent_manager.py` to store intents and provide filtering per region.
   - Implement `SdwanController` with hooks:
     - `update_topology(regions, intents)`
     - `assign_paths(slice_id, source_region, target_region)`
     - `export_policy_overrides()` → fed into `NetworkEnvironment.policy_context`.

2. **Multi-region runner**
   - Build `multi_region.py` that instantiates `NetworkEnvironment` objects, their edges, and WAN links.
   - Step loop:
     - `controller.collect_regional_stats()` (aggregate throughput, SLA violations).
     - `controller.optimise_paths()` (maybe heuristics first, ML later).
     - Push updated contexts into each region and call `env.step(...)`.

3. **RIC API Extensions**
   - Routes: `POST /intents`, `GET /intents`, `DELETE /intents/{id}`.
   - Optional: `POST /sdwan/rebalance` to trigger manual recomputation.
   - Include WAN info in `/metrics` snapshots for visualization.

4. **Telemetry & Visualization**
   - Register SD-WAN metrics in `utils.DEFAULT_TELEMETRY_COLLECTOR`.
   - Add Grafana panels (WAN latency, edge utilization, intent counts) referencing `interface_monitoring.md`.

5. **Documentation & Deliverables**
   - Update `README` with an SD-WAN section and diagram.
   - Compose a short whitepaper (2–3 pages) summarizing architecture + example results.
   - Provide sample notebooks or scripts demonstrating tenant intent scenarios.

## Research Use Cases

1. **QoS-aware path selection**: Show how switching WAN paths reduces SLA violations under congestion.
2. **Tenant intent injection**: Demonstrate dynamic weight tuning when multiple intents compete for WAN resources.
3. **Acceleration-aware orchestration**: Evaluate when DPU-enabled edges provide measurable gains (lower latency, higher throughput) and how the controller decides to use them.

By following this plan, the simulator becomes a realistic SD-WAN controller testbed, suitable for the MSCA research objectives and for showcasing autonomous, intent-driven management across radio-edge and WAN domains.
