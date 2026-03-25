from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .interfaces import CoreInterfaces, N2Message, N3FlowStats


@dataclass
class SessionContext:
    ue_id: int
    slice_id: str
    qos_profile: str
    state: str = "inactive"  # inactive | establishing | active | released
    upf_id: Optional[str] = None
    policy_id: Optional[str] = None


class AMF:
    def __init__(self):
        self.registry: Dict[int, SessionContext] = {}
        self.registration_events = 0

    def register(self, ue_id: int, slice_id: str, qos_profile: str) -> SessionContext:
        ctx = self.registry.get(ue_id)
        if ctx is None:
            ctx = SessionContext(ue_id=ue_id, slice_id=slice_id, qos_profile=qos_profile)
            self.registry[ue_id] = ctx
        ctx.state = "establishing"
        self.registration_events += 1
        return ctx

    def release(self, ue_id: int):
        ctx = self.registry.get(ue_id)
        if ctx:
            ctx.state = "released"

    def active_subscribers(self) -> int:
        return sum(1 for ctx in self.registry.values() if ctx.state == "active")


class PCF:
    def __init__(self):
        self.policies: Dict[str, Dict] = {
            "default": {"qos": "embb", "priority": 1},
            "latency": {"qos": "urllc", "priority": 2},
        }

    def resolve_policy(self, slice_id: str, qos_profile: str) -> Dict:
        if qos_profile == "latency" or slice_id.lower().startswith("urllc"):
            return {"policy_id": "latency", **self.policies["latency"]}
        return {"policy_id": "default", **self.policies["default"]}


class UPF:
    def __init__(self, upf_id: str, capacity_gbps: float = 12.0):
        self.upf_id = upf_id
        self.capacity_gbps = capacity_gbps
        self.current_throughput_gbps = 0.0
        self.user_traffic: Dict[int, float] = {}

    def route_traffic(self, ue_id: int, throughput_mbps: float):
        self.user_traffic[ue_id] = throughput_mbps
        self._recompute()

    def release(self, ue_id: int):
        self.user_traffic.pop(ue_id, None)
        self._recompute()

    def _recompute(self):
        total_mbps = sum(self.user_traffic.values())
        self.current_throughput_gbps = total_mbps / 1000.0

    @property
    def load_ratio(self) -> float:
        return min(self.current_throughput_gbps / max(self.capacity_gbps, 1e-6), 2.0)


class SMF:
    def __init__(self, upf: UPF, pcf: Optional[PCF] = None):
        self.upf = upf
        self.pcf = pcf or PCF()
        self.active_sessions = 0

    def setup_session(self, ctx: SessionContext):
        policy = self.pcf.resolve_policy(ctx.slice_id, ctx.qos_profile)
        ctx.policy_id = policy["policy_id"]
        ctx.upf_id = self.upf.upf_id
        ctx.state = "active"
        self.active_sessions += 1

    def release_session(self, ctx: SessionContext):
        if ctx.state == "active":
            self.active_sessions = max(self.active_sessions - 1, 0)
        ctx.state = "released"
        if ctx.upf_id:
            self.upf.release(ctx.ue_id)


class CoreNetwork:
    def __init__(self, upf_capacity_gbps: float = 12.0):
        self.amf = AMF()
        self.upf = UPF("upf-0", capacity_gbps=upf_capacity_gbps)
        self.smf = SMF(self.upf)
        self.telemetry = None
        self.interfaces = CoreInterfaces()
        self._last_interface_stats = {
            "n2_messages": 0,
            "n3_reports": 0,
            "n2_queue": 0,
            "n3_queue": 0,
        }

    def attach_telemetry(self, collector):
        self.telemetry = collector

    def ensure_session(self, ue_id: int, slice_id: str, qos_profile: str):
        ctx = self.amf.register(ue_id, slice_id, qos_profile)
        self._send_n2_message(
            N2Message(ue_id=ue_id, procedure="registration", status=ctx.state, info={"slice_id": slice_id})
        )
        if ctx.state != "active":
            self.smf.setup_session(ctx)
            self._send_n2_message(
                N2Message(ue_id=ue_id, procedure="session_setup", status="active", info={"policy": ctx.policy_id})
            )
        return ctx

    def update_traffic(self, per_user_throughput: Dict[int, float]):
        for ue_id, throughput in per_user_throughput.items():
            ctx = self.amf.registry.get(ue_id)
            if ctx and ctx.state == "active":
                self.upf.route_traffic(ue_id, throughput)
                self.interfaces.send_n3(
                    N3FlowStats(
                        ue_id=ue_id,
                        throughput_mbps=throughput,
                        packet_loss=0.01 if throughput <= 0 else 0.0,
                    )
                )

    def emit_metrics(self, timestamp: float):
        if not self.telemetry:
            return
        self.telemetry.emit(
            "core.upf_throughput_gbps",
            self.upf.current_throughput_gbps,
            labels={"upf_id": self.upf.upf_id},
            timestamp=timestamp,
        )
        self.telemetry.emit(
            "core.upf_load_ratio",
            self.upf.load_ratio,
            labels={"upf_id": self.upf.upf_id},
            timestamp=timestamp,
        )
        self.telemetry.emit(
            "core.amf_registered_users",
            float(len(self.amf.registry)),
            labels={"component": "amf"},
            timestamp=timestamp,
        )
        self.telemetry.emit(
            "core.smf_active_sessions",
            float(self.smf.active_sessions),
            labels={"component": "smf"},
            timestamp=timestamp,
        )
        if self.telemetry and self._last_interface_stats:
            self.telemetry.emit(
                "interface.n2_queue",
                float(self._last_interface_stats.get("n2_queue", 0)),
                labels={"link": "n2"},
                timestamp=timestamp,
            )
            self.telemetry.emit(
                "interface.n3_queue",
                float(self._last_interface_stats.get("n3_queue", 0)),
                labels={"link": "n3"},
                timestamp=timestamp,
            )
            self.telemetry.emit(
                "interface.n2_messages",
                float(self._last_interface_stats.get("n2_messages", 0)),
                labels={"link": "n2"},
                timestamp=timestamp,
            )
            self.telemetry.emit(
                "interface.n3_reports",
                float(self._last_interface_stats.get("n3_reports", 0)),
                labels={"link": "n3"},
                timestamp=timestamp,
            )

    def service_interfaces(self):
        deliveries = self.interfaces.poll()
        self._last_interface_stats = {
            "n2_messages": len(deliveries.get("n2", [])),
            "n3_reports": len(deliveries.get("n3", [])),
            "n2_queue": self.interfaces.n2.queue_depth(),
            "n3_queue": self.interfaces.n3.queue_depth(),
        }
        return self._last_interface_stats

    def _send_n2_message(self, message: N2Message):
        self.interfaces.send_n2(message)


__all__ = ["CoreNetwork", "SessionContext", "AMF", "SMF", "UPF", "PCF"]


__all__ = ["CoreNetwork", "SessionContext", "AMF", "SMF", "UPF", "PCF"]
