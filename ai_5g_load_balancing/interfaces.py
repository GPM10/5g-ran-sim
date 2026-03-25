from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple


# ------------------------------ Generic link ------------------------------
@dataclass
class InterfaceLink:
    name: str
    latency_ms: float
    jitter_ms: float = 0.0
    drop_rate: float = 0.0
    bandwidth_mbps: float = 1000.0
    _queue: Deque[Tuple[float, Dict]] = field(default_factory=deque)

    def transmit(self, payload: Dict) -> None:
        delay = self.latency_ms + random.uniform(-self.jitter_ms, self.jitter_ms)
        delay = max(delay, 0.0)
        delivery_time = time.time() + delay / 1000.0
        if random.random() < self.drop_rate:
            return
        self._queue.append((delivery_time, payload))

    def receive(self) -> List[Dict]:
        now = time.time()
        ready = []
        while self._queue and self._queue[0][0] <= now:
            _, payload = self._queue.popleft()
            ready.append(payload)
        return ready

    def queue_depth(self) -> int:
        return len(self._queue)


# ------------------------------ E2 interface ------------------------------
@dataclass
class E2KpmReport:
    cell_id: int
    timestamp: float
    metrics: Dict[str, float]


@dataclass
class E2ControlMessage:
    ue_id: int
    target_bs: int
    priority: int = 0


class E2Interface:
    def __init__(
        self,
        control_link: Optional[InterfaceLink] = None,
        report_link: Optional[InterfaceLink] = None,
    ):
        self.control_link = control_link or InterfaceLink(
            "E2-control", latency_ms=5.0, jitter_ms=2.0, drop_rate=0.01
        )
        self.report_link = report_link or InterfaceLink(
            "E2-kpm", latency_ms=10.0, jitter_ms=3.0, drop_rate=0.02
        )
        self.subscriptions: Dict[str, Dict] = {}

    def publish_kpm(self, report: E2KpmReport):
        self.report_link.transmit({"type": "kpm", "report": report})

    def poll_kpm(self) -> List[Dict]:
        return self.report_link.receive()

    def send_control(self, message: E2ControlMessage):
        self.control_link.transmit({"type": "control", "message": message})

    def poll_control(self) -> List[Dict]:
        return self.control_link.receive()

    def add_subscription(self, name: str, filter_args: Dict):
        self.subscriptions[name] = filter_args

    def control_queue_depth(self) -> int:
        return self.control_link.queue_depth()

    def report_queue_depth(self) -> int:
        return self.report_link.queue_depth()


# ------------------------------ F1 interface ------------------------------
@dataclass
class F1SetupMessage:
    cu_id: str
    du_id: str
    payload: Dict


@dataclass
class F1UPlanePacket:
    ue_id: int
    throughput_mbps: float
    latency_ms: float


class F1Interface:
    def __init__(self, name: str, control_link: Optional[InterfaceLink] = None, user_link: Optional[InterfaceLink] = None):
        self.name = name
        self.control = control_link or InterfaceLink(f"{name}-F1C", latency_ms=1.0, jitter_ms=0.2, drop_rate=0.0)
        self.user = user_link or InterfaceLink(f"{name}-F1U", latency_ms=0.5, jitter_ms=0.1, drop_rate=0.0)

    def send_setup(self, msg: F1SetupMessage):
        self.control.transmit({"type": "setup", "msg": msg})

    def send_control(self, payload: Dict):
        self.control.transmit({"type": "control", "payload": payload})

    def send_user_plane(self, packet: F1UPlanePacket):
        self.user.transmit({"type": "uplink", "packet": packet})

    def poll(self) -> Dict[str, List[Dict]]:
        return {
            "control": self.control.receive(),
            "user": self.user.receive(),
        }


# ------------------------------ N2/N3 interfaces ------------------------------
@dataclass
class N2Message:
    ue_id: int
    procedure: str
    status: str
    info: Optional[Dict] = None


@dataclass
class N3FlowStats:
    ue_id: int
    throughput_mbps: float
    packet_loss: float


class CoreInterfaces:
    def __init__(self):
        self.n2 = InterfaceLink("N2", latency_ms=8.0, jitter_ms=3.0, drop_rate=0.01)
        self.n3 = InterfaceLink("N3", latency_ms=3.0, jitter_ms=1.0, drop_rate=0.005)

    def send_n2(self, msg: N2Message):
        self.n2.transmit({"type": "n2", "msg": msg})

    def send_n3(self, stats: N3FlowStats):
        self.n3.transmit({"type": "n3", "stats": stats})

    def poll(self) -> Dict[str, List[Dict]]:
        return {
            "n2": self.n2.receive(),
            "n3": self.n3.receive(),
        }


__all__ = [
    "InterfaceLink",
    "E2Interface",
    "E2KpmReport",
    "E2ControlMessage",
    "F1Interface",
    "F1SetupMessage",
    "F1UPlanePacket",
    "CoreInterfaces",
    "N2Message",
    "N3FlowStats",
]
