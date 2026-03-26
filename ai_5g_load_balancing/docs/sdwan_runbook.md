# SD-WAN Testbed Runbook

This document summarizes the SD-WAN extensions (multi-region orchestration, tenant intents, telemetry) and explains how to reproduce the main scenarios.

## Architecture Overview

```
Tenant Intent API ─┐
                   ├── IntentManager ──► SdwanController
Telemetry (E2/F1) ─┘                         │
                                              ▼
                                  +---------------------------+
                                  |  MultiRegionSimulator     |
                                  |  region{i}:               |
                                  |    NetworkEnvironment     |
                                  |    SdwanEdge (CPU/DPU)    |
                                  +---------------------------+
                                          │ WAN Links
                              WanLink(latency/bw/qos, metrics)
```

### Key Modules
- `sdwan_edge.py`: models regional SD-WAN ingress/egress nodes, with CPU/DPU flags, bandwidth, telemetry hooks.
- `sdwan_wan.py`: configurable WAN links (latency, bandwidth, QoS tiers) with utilization & queue tracking.
- `sdwan_controller.py`: orchestrates regions+links, refreshes telemetry, injects WAN context into each region’s `policy_context`.
- `intent_manager.py`: stores tenant intents (slice, latency, bandwidth, priority) and exposes policy dictionaries.
- `multi_region.py`: CLI entry point that spins up multiple `NetworkEnvironment`s plus edges/links and steps them in lockstep.
- `ric_api.py`: `/intents` endpoints (GET/POST/DELETE) feed intents into the controller; snapshots embed intent state.

## Running Multi-Region Simulations

```
(.venv) python -m ai_5g_load_balancing.multi_region --regions 3 --steps 50 --use-dpu
```

Flags:
- `--regions N`: number of independent environments. Defaults to 2 (minimum 2).
- `--steps N`: simulation steps per region.
- `--use-dpu`: alternate regions use DPU-accelerated edges (higher bandwidth/CPU capacity). Without this flag all edges are CPU-only.

The command prints a per-region summary (throughput, latency, edge utilization). Additional metrics appear on the Prometheus endpoint (if enabled via `--metrics-exporter prometheus` when running individual regions).

## Injecting Tenant Intents

1. Run the RIC API (`python -m ai_5g_load_balancing.ric_api`).
2. Submit intents:
   ```bash
   curl -X POST http://localhost:8000/intents -H "Content-Type: application/json" -d '{
     "tenant": "xr-startup",
     "slice_id": "xr",
     "latency_target_ms": 25,
     "bandwidth_mbps": 500,
     "priority": 3
   }'
   ```
3. List/remove intents:
   - `GET /intents`
   - `DELETE /intents/{intent_id}`

The `IntentManager` synchronizes with every `NetworkEnvironment`, so association heuristics receive `context["intents"]`. Higher-priority intents boost scoring for UEs in that slice, while WAN context reflects per-link latency/utilization to influence decisions.

## Telemetry & Dashboards

New Prometheus metrics (see `utils.DEFAULT_TELEMETRY_COLLECTOR`):
- `sdwan.intent_count`
- `sdwan.edge_cpu_utilization`
- `sdwan.edge_link_utilization`
- `sdwan.wan_link_latency_ms`
- `sdwan.wan_link_utilization`

Existing RAN metrics (`radio.*`, `interface.f1_*`, `core.*`) remain available, allowing unified dashboards that correlate WAN behavior with RAN performance. Suggested panels:
1. WAN link utilization/latency per path.
2. Edge CPU/DPU utilization vs. tenant demand.
3. Intent count and per-slice SLA compliance (latency buckets).

## Reproducing Intent-Driven Scenarios

1. Start a multi-region simulation (with Prometheus exporter enabled via `--metrics-exporter prometheus` in each region or by running RIC API + region loops).
2. Submit multiple intents (e.g., `slice_id` “xr” and “mec”) with different priorities/latency targets.
3. Observe:
   - `/intents` responses showing active intents.
   - Controller logs (`sdwan_controller.py`) when WAN metrics change (edge utilization, link latency).
   - Grafana panels reflecting `sdwan.*` metrics; check that high-priority intents reduce latency by rerouting over premium links (if DPUs are available).

## Notes & Next Steps

- **WAN Topology**: `multi_region.py` currently constructs full-mesh links. Extend `SdwanController.add_link()` usage to build arbitrary topologies or dynamic failures.
- **ML Loop**: The `policy_context` now contains `f1`, `ru`, `core`, `wan`, and `intents`. Plugging a contextual bandit / RL agent into `SdwanController` would enable zero-touch adjustments.
- **Documentation**: Keep `docs/sdwan_extension_plan.md` for high-level roadmap; this runbook serves as the operational guide.
