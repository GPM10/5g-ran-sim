# Interface Monitoring Guide

This guide explains how the simulator exposes the main 5G interfaces—E2 (RIC ↔ RAN), F1 (CU ↔ DU/RU), and N2/N3 (RAN ↔ Core)—along with the key telemetry and API entry points.

## E2 (RIC ↔ RAN)

*Components*: `E2Interface`, `/metrics` snapshot, `/e2` endpoint.

*Behavior*:
- Control actions sent via `/actions` are enqueued as `E2ControlMessage` objects with simulated latency, jitter, and drop rate.
- KPI reports are emitted per cell after each step as `E2KpmReport` and travel over a separate link with its own latency profile.

*APIs*:
- `GET /e2` returns queue depth for the control and KPI channels plus the last 50 delivered reports.
- `GET /metrics` includes the latest KPM snapshots (`snapshot["e2"]`) so xApps can react to transport congestion.

*Telemetry*:
- The `/e2` payload embedded in `/metrics` and `/e2` includes `control_queue_depth`, `kpm_queue_depth`, and recent reports. You can forward these to Prometheus (e.g., small exporter scraping `/e2`) to visualize queue growth during stress tests.

## F1 (CU ↔ DU/RU)

*Components*: `F1Interface` per DU, `RadioUnit` objects, telemetry metrics.

*Behavior*:
- Each handover enqueues an F1-C control payload; each UE throughput sample produces an F1-U packet that must traverse the link before the DU sees it.
- Queue depth reflects fronthaul latency and capacity limits; packet drops reflect link drop rates.

*Metrics* (Prometheus names):
- `interface.f1_control_queue`, `interface.f1_user_queue` – current queue depth per DU.
- `interface.f1_control_packets`, `interface.f1_user_packets` – packets delivered in the last step.
- `radio.ru_load_ratio`, `radio.ru_temperature_c` – RU-side RF utilization and thermal behavior.

*Dashboards*:
- Combine RU metrics with cell load and DU utilization to spot fronthaul bottlenecks.
- Alert if `interface.f1_user_queue` grows steadily (possible fronthaul congestion) or if RU temperatures exceed 80 °C.

## N2/N3 (RAN ↔ 5G Core)

*Components*: `CoreNetwork.interfaces`, `CoreInterfaces`, `/core` endpoint.

*Behavior*:
- Every UE registration/session setup generates N2 signaling messages; per-UE throughput samples send N3 flow reports.
- The core polls these interfaces to account for latency/jitter before updating telemetry.

*Metrics*:
- `interface.n2_queue`, `interface.n3_queue` – queued signaling/flow reports.
- `interface.n2_messages`, `interface.n3_reports` – delivered messages per simulation step.
- Existing core KPIs (`core.upf_load_ratio`, `core.smf_active_sessions`, etc.) remain in `/metrics`.

*API*:
- `GET /core` returns UPF load, AMF registration counts, and SMF sessions for quick inspection or automated policy hooks.

## Workflow Summary

1. **Run the simulator**: `python -m ai_5g_load_balancing.main --steps 300 --metrics-exporter prometheus`.
2. **Monitor metrics**: point Prometheus/Grafana at `http://localhost:9102/metrics` and add panels for the `interface.*`, `radio.ru_*`, and `core.*` KPIs listed above.
3. **Inspect interfaces**: use `curl http://localhost:8000/e2` / `/core` (if `ric_api` is running) to view transport state alongside snapshot data.
4. **Stress test**: send bursts of `/actions` requests or scale demand in the simulator to observe queue buildup and verify telemetry/alarm thresholds.

These hooks provide end-to-end visibility from the xApp (E2) through fronthaul (F1) down to the core (N2/N3), allowing you to evaluate how control decisions behave when transport links are congested or impaired.
