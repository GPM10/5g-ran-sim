import logging
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


logger = logging.getLogger(__name__)

NOISE_FLOOR_DBM = -104  # thermal noise reference

# Fading configuration defaults
FADING_CONFIG = {
    "los_model": "rician",
    "nlos_model": "rayleigh",
    "rician_k": 6.0,
    "nakagami_m": 1.5,
    "shadowing_sigma_db": 6.0,
    "enable_shadowing": True,
}


def link_distance(ue, bs):
    bs_height = getattr(bs, "height_m", 0.0)
    ue_z = getattr(ue, "z", 0.0)
    ue_height = getattr(ue, "height_m", 1.5) if hasattr(ue, "height_m") else 1.5
    x_term = (ue.x - bs.x) ** 2
    y_term = (ue.y - bs.y) ** 2
    z_term = (ue_z + ue_height - (bs.z + bs_height)) ** 2
    return np.sqrt(x_term + y_term + z_term)


def los_probability(distance_m, tier, environment=None):
    distance_m = max(distance_m, 1.0)
    if tier == "micro":
        base = 0.9 if distance_m <= 18 else max(0.1, np.exp(-(distance_m - 18) / 36))
    else:
        base = 0.95 if distance_m <= 30 else max(
            0.05, np.exp(-(distance_m - 30) / 80)
        )
    if environment == "indoor":
        base *= 0.6
    return float(np.clip(base, 0.01, 0.99))


def _legacy_path_loss(distance_m, params, is_los):
    exp = params["los_exp"] if is_los else params["nlos_exp"]
    intercept = params["los_intercept"] if is_los else params["nlos_intercept"]
    distance_m = max(distance_m, 1.0)
    return intercept + 10 * exp * np.log10(distance_m)


def _free_space_loss(distance_m, frequency_ghz, **_):
    distance_km = max(distance_m / 1000.0, 1e-3)
    freq_mhz = max(frequency_ghz * 1000.0, 1.0)
    return 32.45 + 20.0 * np.log10(distance_km) + 20.0 * np.log10(freq_mhz)


def _cost231_loss(distance_m, frequency_ghz, bs, ue_height=1.5, **kwargs):
    params = kwargs.get("params", {})
    distance_km = max(distance_m / 1000.0, 1e-3)
    freq_mhz = max(frequency_ghz * 1000.0, 150.0)
    hb = max(getattr(bs, "height_m", 25.0), 5.0)
    hm = max(ue_height, 1.0)
    c_env = params.get("environment_correction_db", 3.0)
    a_hm = (1.1 * np.log10(freq_mhz) - 0.7) * hm - (1.56 * np.log10(freq_mhz) - 0.8)
    return (
        46.3
        + 33.9 * np.log10(freq_mhz)
        - 13.82 * np.log10(hb)
        - a_hm
        + (44.9 - 6.55 * np.log10(hb)) * np.log10(distance_km)
        + c_env
    )


def _nyu_mmwave_loss(distance_m, frequency_ghz, is_los=True, **kwargs):
    params = kwargs.get("params", {})
    n_los = params.get("los_exp", 2.0)
    n_nlos = params.get("nlos_exp", 3.2)
    sigma_los = params.get("los_sigma_db", 4.0)
    sigma_nlos = params.get("nlos_sigma_db", 7.0)
    intercept = 32.4
    path_exp = n_los if is_los else n_nlos
    sigma = sigma_los if is_los else sigma_nlos
    distance_m = max(distance_m, 1.0)
    return (
        intercept
        + 10 * path_exp * np.log10(distance_m)
        + 20 * np.log10(max(frequency_ghz, 0.1))
        + np.random.normal(0.0, sigma)
    )


PATH_LOSS_MODELS = {
    "free_space": _free_space_loss,
    "cost231": _cost231_loss,
    "nyu_mmwave": _nyu_mmwave_loss,
}


def path_loss_db(distance_m, bs, carrier, ue_height=1.5, is_los=True):
    params = carrier.path_loss_params or bs.path_loss_params or {}
    model_name = (carrier.path_loss_model or bs.path_loss_model or "free_space").lower()
    if {"los_exp", "nlos_exp", "los_intercept", "nlos_intercept"} <= params.keys():
        return _legacy_path_loss(distance_m, params, is_los=is_los)
    model_fn = PATH_LOSS_MODELS.get(model_name, _free_space_loss)
    return model_fn(
        distance_m=distance_m,
        frequency_ghz=getattr(carrier, "frequency_ghz", 3.5),
        bs=bs,
        ue_height=ue_height,
        params=params,
        is_los=is_los,
    )


def received_power_dbm(ue, bs, carrier, is_los=True):
    dist = link_distance(ue, bs)
    ue_height = getattr(ue, "height_m", 1.5)
    loss = path_loss_db(dist, bs, carrier, ue_height=ue_height, is_los=is_los)
    return carrier.tx_power_dbm - loss


def signal_strength(ue, bs):
    """Return expected LOS received power in dBm for the BS primary carrier."""
    carrier = bs.primary_carrier
    return received_power_dbm(ue, bs, carrier, is_los=True)


def dbm_to_mw(dbm):
    return 10 ** (dbm / 10)


def sample_rayleigh_power():
    sigma = 1 / np.sqrt(2)
    amplitude = np.random.rayleigh(scale=sigma)
    return amplitude**2


def sample_rician_power(k_factor):
    k_factor = max(k_factor, 0.0)
    sigma = np.sqrt(1 / (2 * (k_factor + 1)))
    los = np.sqrt(k_factor / (k_factor + 1))
    x = np.random.normal(los, sigma)
    y = np.random.normal(0.0, sigma)
    amplitude = np.sqrt(x**2 + y**2)
    return amplitude**2


def sample_nakagami_power(m, omega=1.0):
    m = max(m, 0.1)
    return np.random.gamma(shape=m, scale=omega / m)


def sample_shadowing_gain(sigma_db):
    sigma_db = max(sigma_db, 0.0)
    offset_db = np.random.normal(0.0, sigma_db)
    return 10 ** (offset_db / 10)


def sample_fading_gain_link(is_los, config=None):
    cfg = config if config else FADING_CONFIG
    model = cfg.get("los_model" if is_los else "nlos_model", "rayleigh").lower()
    if model == "rician":
        fast = sample_rician_power(cfg.get("rician_k", 4.0))
    elif model == "nakagami":
        fast = sample_nakagami_power(cfg.get("nakagami_m", 1.5))
    else:
        fast = sample_rayleigh_power()

    if cfg.get("enable_shadowing", True):
        fast *= sample_shadowing_gain(cfg.get("shadowing_sigma_db", 6.0))
    return fast


def _match_carrier(other_bs, carrier):
    match = other_bs.get_carrier(carrier.name)
    if match:
        return match
    for cand in other_bs.iter_carriers():
        if abs(cand.frequency_ghz - carrier.frequency_ghz) <= 0.25:
            return cand
    return None


def sinr_linear(ue, serving_bs, base_stations, carrier, fading_config=None):
    d_serv = link_distance(ue, serving_bs)
    los_serv = np.random.rand() < los_probability(
        d_serv, serving_bs.tier, environment=getattr(ue, "environment", None)
    )
    signal_dbm = received_power_dbm(ue, serving_bs, carrier, is_los=los_serv)
    signal_mw = dbm_to_mw(signal_dbm) * sample_fading_gain_link(
        los_serv, fading_config
    )

    interference_mw = 0.0
    for other in base_stations:
        if other.bs_id == serving_bs.bs_id:
            continue
        other_carrier = _match_carrier(other, carrier)
        if other_carrier is None:
            continue
        d_int = link_distance(ue, other)
        los_int = np.random.rand() < los_probability(
            d_int, other.tier, environment=getattr(ue, "environment", None)
        )
        power_dbm = received_power_dbm(ue, other, other_carrier, is_los=los_int)
        interference_mw += dbm_to_mw(power_dbm) * sample_fading_gain_link(
            los_int, fading_config
        )

    noise_mw = dbm_to_mw(NOISE_FLOOR_DBM)
    denominator = noise_mw + interference_mw
    if denominator <= 0:
        denominator = noise_mw
    return signal_mw / denominator


def shannon_throughput_mbps(sinr_value, bandwidth_mhz):
    spectral_eff = np.log2(1 + sinr_value)
    return (bandwidth_mhz * 1e6 * spectral_eff) / 1e6


CQI_THRESHOLDS_DB = [
    -6.7,
    -4.7,
    -2.3,
    0.0,
    1.0,
    3.0,
    5.0,
    7.0,
    9.0,
    11.0,
    12.7,
    14.7,
    16.7,
    18.7,
    21.0,
]

MCS_EFFICIENCY = [
    0.1523,
    0.2344,
    0.3770,
    0.6016,
    0.8770,
    1.1758,
    1.4766,
    1.9141,
    2.4063,
    2.7305,
    3.3223,
    3.9023,
    4.5234,
    5.1152,
    5.5547,
    5.8984,
]


def sinr_to_cqi(sinr_db):
    for idx, threshold in enumerate(CQI_THRESHOLDS_DB):
        if sinr_db < threshold:
            return max(idx, 0)
    return len(CQI_THRESHOLDS_DB)


def cqi_to_efficiency(cqi_idx):
    cqi_idx = int(np.clip(cqi_idx, 0, len(MCS_EFFICIENCY) - 1))
    return MCS_EFFICIENCY[cqi_idx]


def phy_rate_from_rbs(bandwidth_mhz, resource_blocks, cqi_efficiency, allocated_rbs):
    if allocated_rbs <= 0:
        return 0.0
    total_rbs = max(resource_blocks, 1)
    rb_bw_hz = (bandwidth_mhz * 1e6) / total_rbs
    rb_rate_mbps = cqi_efficiency * rb_bw_hz / 1e6
    return rb_rate_mbps * allocated_rbs


def user_throughput_from_phy(
    ue,
    bandwidth_mhz,
    resource_blocks,
    sinr_val,
    allocated_rbs,
    timestep_s=1.0,
):
    sinr_db = 10 * np.log10(sinr_val + 1e-9)
    cqi = sinr_to_cqi(sinr_db)
    eff = cqi_to_efficiency(cqi)
    phy_rate = phy_rate_from_rbs(bandwidth_mhz, resource_blocks, eff, allocated_rbs)
    backlog_rate = ue.backlog_mbits / max(timestep_s, 1e-6)
    desired_rate = max(backlog_rate, ue.demand)
    served_rate = min(phy_rate, desired_rate)
    served_data = served_rate * timestep_s
    ue.backlog_mbits = max(ue.backlog_mbits - served_data, 0.0)
    return served_rate, phy_rate, cqi


def estimate_latency_ms(ue, throughput_mbps, backlog_before_mbits, timestep_s=1.0):
    if throughput_mbps <= 1e-6:
        return ue.latency_budget_ms * 5
    wait_seconds = backlog_before_mbits / max(throughput_mbps, 1e-3)
    latency = min(wait_seconds * 1000.0, ue.latency_budget_ms * 5)
    return latency


# ---------------------------------------------------------------------------
# Telemetry & Monitoring infrastructure
# ---------------------------------------------------------------------------


@dataclass
class TelemetryThreshold:
    warn: Optional[float] = None
    crit: Optional[float] = None
    comparison: str = "gte"  # "gte" or "lte"

    def evaluate(self, value: float) -> Optional[str]:
        if self.warn is None and self.crit is None:
            return None
        if self.comparison == "gte":
            if self.crit is not None and value >= self.crit:
                return "critical"
            if self.warn is not None and value >= self.warn:
                return "warning"
        else:
            if self.crit is not None and value <= self.crit:
                return "critical"
            if self.warn is not None and value <= self.warn:
                return "warning"
        return None


@dataclass
class MetricSample:
    value: float
    labels: Dict[str, Any]
    timestamp: float


@dataclass
class TelemetryMetric:
    name: str
    description: str
    metric_type: str = "gauge"  # gauge | counter
    unit: str = ""
    thresholds: Optional[TelemetryThreshold] = None
    history_size: int = 180
    history: deque = field(
        default_factory=lambda: deque(maxlen=180)
    )  # sliding window for dashboards

    def add_sample(self, sample: MetricSample):
        self.history.append(sample)

    def latest_value(self) -> Optional[MetricSample]:
        return self.history[-1] if self.history else None

    def check_threshold(self, sample: MetricSample) -> Optional[str]:
        if self.thresholds is None:
            return None
        return self.thresholds.evaluate(sample.value)


class TelemetryExporter:
    def emit(
        self, metric: TelemetryMetric, sample: MetricSample, severity: Optional[str]
    ):
        raise NotImplementedError


class PrometheusExporter(TelemetryExporter):
    def __init__(self, port: int = 9102, namespace: str = "ai5g", start_server: bool = True):
        try:
            from prometheus_client import CollectorRegistry, Counter, Gauge, start_http_server
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "prometheus_client is required for PrometheusExporter"
            ) from exc
        self._Counter = Counter
        self._Gauge = Gauge
        self.registry = CollectorRegistry()
        self.namespace = namespace
        self.metrics: Dict[Tuple[str, Tuple[str, ...]], Any] = {}
        if start_server:
            start_http_server(port, registry=self.registry)

    def _metric_key(self, metric: TelemetryMetric, labels: Dict[str, Any]):
        label_names = tuple(sorted(labels.keys()))
        return (metric.name, label_names)

    def _resolve_metric(self, metric: TelemetryMetric, labels: Dict[str, Any]):
        key = self._metric_key(metric, labels)
        if key not in self.metrics:
            label_names = key[1]
            params = {
                "name": metric.name.replace(".", "_"),
                "documentation": metric.description,
                "labelnames": label_names,
                "registry": self.registry,
                "namespace": self.namespace,
                "unit": metric.unit or None,
            }
            if metric.metric_type == "counter":
                self.metrics[key] = self._Counter(**params)
            else:
                self.metrics[key] = self._Gauge(**params)
        label_names = key[1]
        prom_metric = self.metrics[key]
        if not label_names:
            return prom_metric
        label_values = [labels[name] for name in label_names]
        return prom_metric.labels(*label_values)

    def emit(self, metric: TelemetryMetric, sample: MetricSample, severity: Optional[str]):
        prom_metric = self._resolve_metric(metric, sample.labels)
        if metric.metric_type == "counter":
            prom_metric.inc(sample.value)
        else:
            prom_metric.set(sample.value)


class OTLPExporter(TelemetryExporter):
    def __init__(
        self,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        timeout_s: float = 2.0,
        raise_on_error: bool = False,
    ):
        import urllib.request

        self.endpoint = endpoint
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout_s = timeout_s
        self.raise_on_error = raise_on_error
        self._urllib = urllib.request

    def emit(self, metric: TelemetryMetric, sample: MetricSample, severity: Optional[str]):
        import json

        payload = {
            "resource_metrics": [
                {
                    "resource": {"attributes": [{"key": "service.name", "value": "ai5g-controller"}]},
                    "scope_metrics": [
                        {
                            "metrics": [
                                {
                                    "name": metric.name,
                                    "description": metric.description,
                                    "unit": metric.unit,
                                    "type": metric.metric_type,
                                    "value": sample.value,
                                    "labels": sample.labels,
                                    "timestamp": sample.timestamp,
                                    "severity": severity,
                                }
                            ]
                        }
                    ],
                }
            ]
        }
        data = json.dumps(payload).encode("utf-8")
        req = self._urllib.Request(self.endpoint, data=data, headers=self.headers, method="POST")
        try:
            self._urllib.urlopen(req, timeout=self.timeout_s)
        except Exception as exc:  # pragma: no cover - network errors
            logger.debug("OTLP export failed for %s: %s", metric.name, exc)
            if self.raise_on_error:
                raise


class TelemetryCollector:
    def __init__(self):
        self.metrics: Dict[str, TelemetryMetric] = {}
        self.exporters: Sequence[TelemetryExporter] = ()
        self._lock = threading.Lock()

    def set_exporters(self, exporters: Sequence[TelemetryExporter]):
        self.exporters = exporters

    def register_metric(
        self,
        name: str,
        description: str,
        metric_type: str = "gauge",
        unit: str = "",
        thresholds: Optional[TelemetryThreshold] = None,
        history_size: int = 180,
    ) -> TelemetryMetric:
        with self._lock:
            metric = self.metrics.get(name)
            if metric:
                return metric
            metric = TelemetryMetric(
                name=name,
                description=description,
                metric_type=metric_type,
                unit=unit,
                thresholds=thresholds,
                history_size=history_size,
                history=deque(maxlen=history_size),
            )
            self.metrics[name] = metric
            return metric

    def emit(
        self, name: str, value: float, labels: Optional[Dict[str, Any]] = None, timestamp: Optional[float] = None
    ) -> Optional[str]:
        with self._lock:
            metric = self.metrics.get(name)
            if metric is None:
                metric = self.register_metric(name, description=name)
        sample = MetricSample(
            value=float(value),
            labels=labels or {},
            timestamp=timestamp if timestamp is not None else time.time(),
        )
        metric.add_sample(sample)
        severity = metric.check_threshold(sample)
        for exporter in self.exporters:
            try:
                exporter.emit(metric, sample, severity)
            except Exception as exc:  # pragma: no cover - exporter errors
                logger.debug("Telemetry exporter %s failed: %s", exporter.__class__.__name__, exc)
        return severity

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            snapshot = {}
            for name, metric in self.metrics.items():
                latest = metric.latest_value()
                snapshot[name] = {
                    "description": metric.description,
                    "unit": metric.unit,
                    "metric_type": metric.metric_type,
                    "latest": {
                        "value": latest.value if latest else None,
                        "labels": latest.labels if latest else {},
                        "timestamp": latest.timestamp if latest else None,
                    },
                }
        return snapshot


DEFAULT_ALERT_THRESHOLDS = {
    "radio.cell_load_ratio": TelemetryThreshold(warn=1.0, crit=1.2, comparison="gte"),
    "radio.ru_load_ratio": TelemetryThreshold(warn=1.0, crit=1.3, comparison="gte"),
    "radio.ru_temperature_c": TelemetryThreshold(warn=70.0, crit=85.0, comparison="gte"),
    "network.avg_throughput_mbps": TelemetryThreshold(warn=8.0, crit=6.0, comparison="lte"),
    "network.avg_latency_ms": TelemetryThreshold(warn=120.0, crit=160.0, comparison="gte"),
    "network.queue_backlog_mbits": TelemetryThreshold(warn=8.0, crit=12.0, comparison="gte"),
    "mobility.handover_success_ratio": TelemetryThreshold(warn=0.85, crit=0.7, comparison="lte"),
    "core.du_processing_utilization": TelemetryThreshold(warn=0.8, crit=1.0, comparison="gte"),
    "core.du_fronthaul_utilization": TelemetryThreshold(warn=0.8, crit=1.0, comparison="gte"),
    "core.cu_cpu_utilization": TelemetryThreshold(warn=0.75, crit=0.95, comparison="gte"),
    "core.upf_load_ratio": TelemetryThreshold(warn=0.8, crit=1.0, comparison="gte"),
}


def _bootstrap_default_collector() -> TelemetryCollector:
    collector = TelemetryCollector()
    collector.register_metric(
        "radio.cell_load_ratio",
        "Current load ratio per base station",
        metric_type="gauge",
        unit="ratio",
        thresholds=DEFAULT_ALERT_THRESHOLDS["radio.cell_load_ratio"],
    )
    collector.register_metric(
        "radio.ru_load_ratio",
        "Radio Unit load ratio",
        metric_type="gauge",
        unit="ratio",
        thresholds=DEFAULT_ALERT_THRESHOLDS["radio.ru_load_ratio"],
    )
    collector.register_metric(
        "radio.ru_temperature_c",
        "Radio Unit temperature (C)",
        metric_type="gauge",
        unit="celsius",
        thresholds=DEFAULT_ALERT_THRESHOLDS["radio.ru_temperature_c"],
    )
    collector.register_metric(
        "network.avg_throughput_mbps",
        "Average UE throughput (Mbps)",
        metric_type="gauge",
        unit="mbps",
        thresholds=DEFAULT_ALERT_THRESHOLDS["network.avg_throughput_mbps"],
    )
    collector.register_metric(
        "network.avg_latency_ms",
        "Average UE latency (ms)",
        metric_type="gauge",
        unit="ms",
        thresholds=DEFAULT_ALERT_THRESHOLDS["network.avg_latency_ms"],
    )
    collector.register_metric(
        "network.queue_backlog_mbits",
        "Average UE queue backlog (Mbits)",
        metric_type="gauge",
        unit="mbits",
        thresholds=DEFAULT_ALERT_THRESHOLDS["network.queue_backlog_mbits"],
    )
    collector.register_metric(
        "mobility.handover_attempts_total",
        "Number of handover attempts in the latest step",
        metric_type="counter",
        unit="events",
    )
    collector.register_metric(
        "mobility.handover_success_ratio",
        "Ratio of successful handovers per step",
        metric_type="gauge",
        unit="ratio",
        thresholds=DEFAULT_ALERT_THRESHOLDS["mobility.handover_success_ratio"],
    )
    collector.register_metric(
        "mobility.handover_failure_total",
        "Handover failures detected per step",
        metric_type="counter",
        unit="events",
    )
    collector.register_metric(
        "network.overloaded_cells",
        "Number of base stations operating above safe load",
        metric_type="gauge",
        unit="count",
    )
    collector.register_metric(
        "core.du_processing_utilization",
        "DU processing utilization ratio",
        metric_type="gauge",
        unit="ratio",
        thresholds=DEFAULT_ALERT_THRESHOLDS["core.du_processing_utilization"],
    )
    collector.register_metric(
        "core.du_fronthaul_utilization",
        "DU fronthaul utilization ratio",
        metric_type="gauge",
        unit="ratio",
        thresholds=DEFAULT_ALERT_THRESHOLDS["core.du_fronthaul_utilization"],
    )
    collector.register_metric(
        "core.cu_cpu_utilization",
        "CU processing utilization ratio",
        metric_type="gauge",
        unit="ratio",
        thresholds=DEFAULT_ALERT_THRESHOLDS["core.cu_cpu_utilization"],
    )
    collector.register_metric(
        "core.upf_throughput_gbps",
        "UPF throughput (Gbps)",
        metric_type="gauge",
        unit="gbps",
    )
    collector.register_metric(
        "core.upf_load_ratio",
        "UPF load ratio",
        metric_type="gauge",
        unit="ratio",
        thresholds=DEFAULT_ALERT_THRESHOLDS["core.upf_load_ratio"],
    )
    collector.register_metric(
        "core.amf_registered_users",
        "Registered subscribers in AMF",
        metric_type="gauge",
        unit="count",
    )
    collector.register_metric(
        "core.smf_active_sessions",
        "Active sessions handled by SMF",
        metric_type="gauge",
        unit="count",
    )
    collector.register_metric(
        "interface.f1_control_queue",
        "F1 control-plane queue depth",
        metric_type="gauge",
        unit="packets",
    )
    collector.register_metric(
        "interface.f1_user_queue",
        "F1 user-plane queue depth",
        metric_type="gauge",
        unit="packets",
    )
    collector.register_metric(
        "interface.f1_control_packets",
        "Delivered F1 control packets per step",
        metric_type="gauge",
        unit="packets",
    )
    collector.register_metric(
        "interface.f1_user_packets",
        "Delivered F1 user-plane packets per step",
        metric_type="gauge",
        unit="packets",
    )
    collector.register_metric(
        "interface.n2_queue",
        "N2 interface queue depth",
        metric_type="gauge",
        unit="messages",
    )
    collector.register_metric(
        "interface.n3_queue",
        "N3 interface queue depth",
        metric_type="gauge",
        unit="messages",
    )
    collector.register_metric(
        "interface.n2_messages",
        "Delivered N2 signaling messages per step",
        metric_type="gauge",
        unit="messages",
    )
    collector.register_metric(
        "interface.n3_reports",
        "Delivered N3 flow reports per step",
        metric_type="gauge",
        unit="messages",
    )
    return collector


DEFAULT_TELEMETRY_COLLECTOR = _bootstrap_default_collector()


def get_dashboard_spec() -> Dict[str, Any]:
    """Return a minimal dashboard configuration consumers can render."""
    return {
        "title": "AI 5G Load-balancing KPIs",
        "panels": [
            {
                "title": "Avg Throughput (Mbps)",
                "metric": "network.avg_throughput_mbps",
                "thresholds": {"warning": 8.0, "critical": 6.0},
            },
            {
                "title": "Avg Latency (ms)",
                "metric": "network.avg_latency_ms",
                "thresholds": {"warning": 120.0, "critical": 160.0},
            },
            {
                "title": "Cell Load Heatmap",
                "metric": "radio.cell_load_ratio",
                "thresholds": {"warning": 1.0, "critical": 1.2},
                "breakdown": "bs_id",
            },
            {
                "title": "Handover Success",
                "metric": "mobility.handover_success_ratio",
                "thresholds": {"warning": 0.85, "critical": 0.7},
            },
            {
                "title": "Queue Backlog",
                "metric": "network.queue_backlog_mbits",
                "thresholds": {"warning": 8.0, "critical": 12.0},
            },
        ],
        "alerts": [
            {
                "name": "Cell Overload",
                "condition": "radio.cell_load_ratio >= 1.2 for 2m",
                "severity": "critical",
            },
            {
                "name": "Throughput Regression",
                "condition": "network.avg_throughput_mbps <= 6.0 for 5m",
                "severity": "critical",
            },
            {
                "name": "Latency Spike",
                "condition": "network.avg_latency_ms >= 160 for 3m",
                "severity": "warning",
            },
            {
                "name": "Handover Failures",
                "condition": "mobility.handover_success_ratio <= 0.7 for 1m",
                "severity": "critical",
            },
        ],
    }
