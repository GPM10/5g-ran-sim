# Telemetry, Monitoring & Alerting

This guide describes how to consume the runtime KPIs exposed by `NetworkEnvironment` through the shared telemetry collector in `utils.py`.

## Metrics

| Metric | Type | Labels | Description | Warning | Critical |
| --- | --- | --- | --- | --- | --- |
| `radio.cell_load_ratio` | gauge | `bs_id`, `tier` | Instantaneous load per base station | 1.0 | 1.2 |
| `network.avg_throughput_mbps` | gauge | `scope` | Mean UE throughput each step | 8 Mbps↓ | 6 Mbps↓ |
| `network.avg_latency_ms` | gauge | `scope` | Mean latency | 120 ms↑ | 160 ms↑ |
| `network.queue_backlog_mbits` | gauge | `scope` | Avg UE backlog | 8 Mbits↑ | 12 Mbits↑ |
| `network.overloaded_cells` | gauge | `scope`, `total_bs` | Count of BS with load ≥ 1.0 | 1 cell | 2+ cells |
| `mobility.handover_attempts_total` | counter | `scope` | Handovers attempted in last step | – | – |
| `mobility.handover_failure_total` | counter | `scope` | Estimated failed handovers per step | 2 | 4 |
| `mobility.handover_success_ratio` | gauge | `scope` | Successful handovers / attempts | 0.85↓ | 0.70↓ |

Thresholds map to alerting defaults (reuse or override in config).

## Exporters

### Prometheus
```python
from utils import DEFAULT_TELEMETRY_COLLECTOR, PrometheusExporter

exporter = PrometheusExporter(port=9102, namespace="ai5g")
DEFAULT_TELEMETRY_COLLECTOR.set_exporters((exporter,))
```
This spins up `/metrics` at `:9102`. Attach Grafana panels directly to the metrics in the table above.

### OTLP/HTTP
```python
from utils import DEFAULT_TELEMETRY_COLLECTOR, OTLPExporter

otlp = OTLPExporter(endpoint="https://otel-collector.example/v1/metrics")
DEFAULT_TELEMETRY_COLLECTOR.set_exporters((otlp,))
```
The exporter ships minimal OTLP-compatible JSON envelopes, suitable for collectors that accept HTTP/JSON ingestion.

## Dashboard Layout

Reference layout returned by `utils.get_dashboard_spec()`:

1. **Throughput trend**: line chart of `network.avg_throughput_mbps` vs. time.
2. **Latency tiles**: line chart plus single-stat for `network.avg_latency_ms`.
3. **Cell load heatmap**: panel keyed by `radio.cell_load_ratio{bs_id=...}`.
4. **Handover success**: area chart of `mobility.handover_success_ratio` with alert band.
5. **Queue backlog**: histogram/line for `network.queue_backlog_mbits`.

## Alert Rules

Example PromQL alert pseudo-code (5-minute windows):

- `avg_over_time(network_avg_latency_ms[5m]) >= 160` → Latency Spike (warning).
- `avg_over_time(network_avg_throughput_mbps[5m]) <= 6` → Throughput Regression (critical).
- `max_over_time(radio_cell_load_ratio[2m]) >= 1.2` → Cell Overload (critical).
- `avg_over_time(mobility_handover_success_ratio[1m]) <= 0.7` → Handover Failure Storm (critical).

These align with the default thresholds baked into the collector to keep the RIC operations center informed about SLA pressure in near real time.
