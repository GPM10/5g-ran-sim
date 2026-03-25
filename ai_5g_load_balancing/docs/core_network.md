# Core Network Simulation

This simulator now models key 5G Core entities to study end-to-end effects between the RAN, CU/DU layers, and the control/user plane.

## Components

- **AMF**: registers UEs, tracks session state, and exposes registration counts for telemetry.
- **SMF**: orchestrates session establishment, queries the PCF for QoS policies, and binds UEs to the UPF instance.
- **PCF**: simplified policy repository that distinguishes between default eMBB-style service and latency-sensitive URLLC flows.
- **UPF**: tracks per-UE throughput routed through the data plane, computes aggregate throughput (Gbps) and load ratios vs. configured capacity.
- **CoreNetwork**: façade that ties the above together, ensuring each UE has an active session, routing throughput reports from the RAN, and emitting telemetry.

## Interaction with the RAN

`NetworkEnvironment` automatically instantiates a `CoreNetwork` unless one is provided. Each simulation step:

1. Calls `CoreNetwork.ensure_session(...)` for every UE, using the UE’s traffic profile and latency budget to select a QoS class.
2. After scheduling, passes per-UE throughput metrics to `CoreNetwork.update_traffic(...)`.
3. Emits telemetry (UPF throughput/load, AMF registrations, SMF active sessions) alongside existing RAN KPIs.

The RIC/xApp surfaces these stats through the `/metrics` snapshot and a dedicated `/core` endpoint, enabling policies to consider core congestion (e.g., UPF load) when deciding handovers.

## Configuration

- Override UPF capacity: `CoreNetwork(upf_capacity_gbps=20.0)`
- Override CU count: set `rl_config["cu_count"]`.
- Extend policies: modify `PCF.policies` or inject a custom PCF via `CoreNetwork`.

## Telemetry Metrics

| Metric | Description |
| --- | --- |
| `core.upf_throughput_gbps` | Real-time throughput through the UPF. |
| `core.upf_load_ratio` | UPF utilization vs. configured capacity. |
| `core.amf_registered_users` | Total UEs with active/establishing sessions. |
| `core.smf_active_sessions` | Number of sessions currently marked active by the SMF. |
| `core.du_*` / `core.cu_*` | Existing CU/DU telemetry from the RAN layer. |

These metrics are exposed via Prometheus/OTLP exporters and can drive alerts or policy inputs. Use them to evaluate how RAN balancing decisions impact the core network and vice versa.
