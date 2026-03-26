import logging
import os
import time
from collections import deque
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Lock
from typing import Deque, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .controller import predictive_mobility_policy
from .environment import NetworkEnvironment
from .intent_manager import IntentManager
from .interfaces import E2ControlMessage, E2Interface, E2KpmReport
from .topology import (
    build_network_from_spec,
    build_reference_network,
    load_topology_spec,
)


LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "ric_api.log"


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("ric_api")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=1_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


logger = _build_logger()


class ActionItem(BaseModel):
    ue_id: int
    target_bs: int


class ActionRequest(BaseModel):
    actions: List[ActionItem]


class IntentPayload(BaseModel):
    tenant: str
    slice_id: str
    latency_target_ms: float
    bandwidth_mbps: float
    priority: int = 1


class SimulationManager:
    def __init__(self, topology_path: Optional[str] = None):
        base_stations, users = self._load_topology(topology_path)
        self.env = NetworkEnvironment(base_stations, users)
        self.e2 = E2Interface()
        self.e2.add_subscription("default-kpm", {"interval_ms": 200})
        self.intent_manager = IntentManager()
        self.env.set_intents(self.intent_manager.as_policy_dict())
        self._bs_id_to_index = {bs.bs_id: idx for idx, bs in enumerate(base_stations)}
        self.states, self.candidate_map = self.env.reset_for_rl()
        self.last_metrics: Optional[Dict[int, Dict[str, float]]] = None
        self.last_snapshot: Optional[Dict] = None
        self.last_timestamp = time.time()
        self.lock = Lock()
        self._recent_kpm: Deque[Dict] = deque(maxlen=50)
        logger.info(
            "Simulation initialized with %d users across %d cells (topology=%s)",
            len(self.env.users),
            len(self.env.base_stations),
            topology_path or "reference",
        )
        if not topology_path:
            logger.info("Using built-in reference topology")

    def _load_topology(self, path: Optional[str]):
        if path:
            topo_file = Path(path)
            try:
                spec = load_topology_spec(topo_file)
                logger.info("Loaded topology spec from %s", topo_file)
                return build_network_from_spec(spec)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to load topology at %s: %s", topo_file, exc)
                logger.warning("Falling back to reference topology")
        return build_reference_network()

    def get_snapshot(self, advance: bool = True):
        with self.lock:
            if self.last_metrics is None or advance:
                self._step()
            else:
                self.last_snapshot = self._build_snapshot()
            return self.last_snapshot

    def apply_actions(self, request: ActionRequest):
        if not request.actions:
            logger.warning("Rejected action request: empty actions list")
            raise HTTPException(status_code=400, detail="Action list is empty")
        for action in request.actions:
            if action.ue_id < 0 or action.ue_id >= len(self.env.users):
                logger.warning("Rejected action: invalid ue_id=%s", action.ue_id)
                raise HTTPException(status_code=400, detail=f"Invalid ue_id {action.ue_id}")
            if action.target_bs not in self._bs_id_to_index:
                logger.warning("Rejected action: invalid target_bs=%s", action.target_bs)
                raise HTTPException(
                    status_code=400, detail=f"Invalid target_bs {action.target_bs}"
                )
            msg = E2ControlMessage(ue_id=action.ue_id, target_bs=action.target_bs)
            self.e2.send_control(msg)
        with self.lock:
            self._step(source="xapp")
            logger.info("Queued %d external actions via E2", len(request.actions))
            return self.last_snapshot

    def get_history(self, tail: int = 50):
        with self.lock:
            history = {}
            for key, values in self.env.history.items():
                history[key] = values[-tail:]
            return history

    def core_state(self) -> Dict:
        core = getattr(self.env, "core_network", None)
        if core is None:
            return {}
        return {
            "upf": {
                "upf_id": core.upf.upf_id,
                "throughput_gbps": core.upf.current_throughput_gbps,
                "load_ratio": core.upf.load_ratio,
                "capacity_gbps": core.upf.capacity_gbps,
            },
            "amf": {
                "registered_users": len(core.amf.registry),
                "active_users": core.amf.active_subscribers(),
                "registrations": core.amf.registration_events,
            },
            "smf": {
                "active_sessions": core.smf.active_sessions,
            },
        }

    def e2_state(self) -> Dict:
        return {
            "control_queue_depth": self.e2.control_queue_depth(),
            "kpm_queue_depth": self.e2.report_queue_depth(),
            "recent_kpm_reports": list(self._recent_kpm),
        }

    def sync_intents(self):
        self.env.set_intents(self.intent_manager.as_policy_dict())

    def list_intents(self):
        with self.lock:
            return [intent.to_dict() for intent in self.intent_manager.list_intents()]

    def create_intent(self, payload: IntentPayload):
        with self.lock:
            intent = self.intent_manager.add_intent(
                tenant=payload.tenant,
                slice_id=payload.slice_id,
                latency_target_ms=payload.latency_target_ms,
                bandwidth_mbps=payload.bandwidth_mbps,
                priority=payload.priority,
            )
            self.sync_intents()
            return intent.to_dict()

    def delete_intent(self, intent_id: int) -> bool:
        with self.lock:
            removed = self.intent_manager.remove_intent(intent_id)
            if removed:
                self.sync_intents()
            return removed

    def _step(self, action_vector: Optional[List[int]] = None, source: Optional[str] = None):
        self._collect_kpm_reports()
        external = action_vector is not None
        if action_vector is None:
            control_mapping = self._drain_control_messages()
            action_vector = self._actions_from_mapping(control_mapping)
        step_source = source or ("xapp" if external else "heuristic")
        states, _, info = self.env.step_with_actions(
            action_vector,
            candidate_map=self.candidate_map,
            update_history=True,
        )
        self.states = states
        self.candidate_map = info["candidate_map"]
        self.last_metrics = info["metrics"]
        self.last_timestamp = time.time()
        self.last_snapshot = self._build_snapshot()
        self._publish_kpm_reports(
            self.last_snapshot["cell_stats"], self.last_snapshot["aggregates"]
        )
        aggregates = self.last_snapshot.get("aggregates", {})
        logger.info(
            "Step via %s controller | avg_latency=%.1f ms | max_load=%.2f | avg_thpt=%.2f Mbps",
            step_source,
            aggregates.get("avg_latency_ms", 0.0),
            aggregates.get("max_cell_load", 0.0),
            aggregates.get("avg_throughput_mbps", 0.0),
        )

    def _heuristic_actions(self) -> List[int]:
        actions = []
        for idx, ue in enumerate(self.env.users):
            best_bs = predictive_mobility_policy(
                ue, self.env.base_stations, context=self.env.policy_context
            )
            actions.append(self._candidate_index(idx, best_bs.bs_id))
        return actions

    def _actions_from_mapping(self, mapping: Dict[int, int]) -> List[int]:
        fallback = self._heuristic_actions()
        actions = []
        for idx, ue in enumerate(self.env.users):
            target_bs = mapping.get(ue.ue_id)
            if target_bs is not None:
                actions.append(self._candidate_index(idx, target_bs))
            else:
                actions.append(fallback[idx])
        return actions

    def _drain_control_messages(self) -> Dict[int, int]:
        mapping: Dict[int, int] = {}
        for payload in self.e2.poll_control():
            message = payload.get("message")
            if isinstance(message, dict):
                message = E2ControlMessage(**message)
            mapping[message.ue_id] = message.target_bs
        return mapping

    def _collect_kpm_reports(self):
        for payload in self.e2.poll_kpm():
            report = payload.get("report")
            if isinstance(report, dict):
                report = E2KpmReport(**report)
            self._recent_kpm.append(
                {
                    "cell_id": report.cell_id,
                    "timestamp": report.timestamp,
                    "metrics": report.metrics,
                }
            )

    def _publish_kpm_reports(self, cell_stats: List[Dict], aggregates: Dict):
        for cell in cell_stats:
            metrics = {
                "load": cell["load"],
                "connected_ues": cell["connected_ues"],
                "capacity_mbps": cell["capacity_mbps"],
                "avg_latency_ms": aggregates.get("avg_latency_ms", 0.0),
                "avg_throughput_mbps": aggregates.get("avg_throughput_mbps", 0.0),
            }
            report = E2KpmReport(
                cell_id=cell["bs_id"],
                timestamp=self.last_timestamp,
                metrics=metrics,
            )
            self.e2.publish_kpm(report)

    def _candidate_index(self, ue_idx: int, bs_id: int) -> int:
        candidates = self.candidate_map[ue_idx]
        if not candidates:
            return 0
        target_idx = self._bs_id_to_index.get(bs_id)
        if target_idx is None:
            return 0
        if target_idx in candidates:
            return candidates.index(target_idx)
        return 0

    def _build_snapshot(self) -> Dict:
        cell_stats = []
        loads = []
        for bs in self.env.base_stations:
            cell_stats.append(
                {
                    "bs_id": bs.bs_id,
                    "tier": bs.tier,
                    "load": float(bs.load),
                    "connected_ues": len(bs.connected_users),
                    "capacity_mbps": bs.capacity_mbps,
                    "x": float(bs.x),
                    "y": float(bs.y),
                    "z": float(bs.z),
                    "height_m": float(bs.height_m),
                    "carriers": [
                        {
                            "name": carrier.name,
                            "frequency_ghz": carrier.frequency_ghz,
                            "bandwidth_mhz": carrier.bandwidth_mhz,
                            "tx_power_dbm": carrier.tx_power_dbm,
                            "capacity_mbps": carrier.capacity_mbps,
                            "resource_blocks": carrier.resource_blocks,
                        }
                        for carrier in bs.iter_carriers()
                    ],
                }
            )
            loads.append(bs.load)

        ue_stats = []
        latencies = []
        throughputs = []
        queues = []
        sinrs = []
        if self.last_metrics is None:
            self.last_metrics = {}
        for ue in self.env.users:
            metrics = self.last_metrics.get(
                ue.ue_id,
                {
                    "throughput": 0.0,
                    "latency": ue.latency_budget_ms,
                    "queue_mbits": ue.backlog_mbits,
                    "cell_load": 0.0,
                    "sinr_db": 0.0,
                },
            )
            ue_stats.append(
                {
                    "ue_id": ue.ue_id,
                    "serving_bs": ue.serving_bs,
                    "traffic_profile": ue.traffic_profile,
                    "latency_ms": float(metrics["latency"]),
                    "throughput_mbps": float(metrics["throughput"]),
                    "queue_mbits": float(metrics.get("queue_mbits", 0.0)),
                    "sinr_db": float(metrics.get("sinr_db", 0.0)),
                    "latency_budget_ms": ue.latency_budget_ms,
                    "x": float(ue.x),
                    "y": float(ue.y),
                    "z": float(getattr(ue, "z", 0.0)),
                    "environment": getattr(ue, "environment", "urban"),
                    "velocity_m_s": float(getattr(ue, "velocity_m_s", 0.0)),
                }
            )
            latencies.append(metrics["latency"])
            throughputs.append(metrics["throughput"])
            queues.append(metrics.get("queue_mbits", 0.0))
            sinrs.append(metrics.get("sinr_db", 0.0))

        snapshot = {
            "timestamp": self.last_timestamp,
            "cell_stats": cell_stats,
            "ue_stats": ue_stats,
            "aggregates": {
                "avg_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
                "avg_throughput_mbps": float(np.mean(throughputs)) if throughputs else 0.0,
                "avg_queue_mbits": float(np.mean(queues)) if queues else 0.0,
                "avg_cell_load": float(np.mean(loads)) if loads else 0.0,
                "max_cell_load": float(np.max(loads)) if loads else 0.0,
                "avg_sinr_db": float(np.mean(sinrs)) if sinrs else 0.0,
            },
            "state_matrix": self.states.tolist(),
            "candidate_map": [
                [self.env.base_stations[idx].bs_id for idx in candidates]
                for candidates in self.candidate_map
            ],
            "demand_event": self.env.last_demand_event,
        }
        snapshot["core"] = self.core_state()
        snapshot["e2"] = self.e2_state()
        snapshot["intents"] = self.intent_manager.as_policy_dict()
        return snapshot

TOPOLOGY_PATH = os.getenv("TOPOLOGY_FILE")

manager = SimulationManager(topology_path=TOPOLOGY_PATH)
app = FastAPI(title="Latency-Aware RIC Metrics API")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "users": len(manager.env.users),
        "cells": len(manager.env.base_stations),
    }


@app.get("/metrics")
def metrics(advance: bool = True):
    return manager.get_snapshot(advance=advance)


@app.post("/actions")
def actions(request: ActionRequest):
    return manager.apply_actions(request)


@app.get("/history")
def history(tail: int = 50):
    return manager.get_history(tail=tail)


@app.get("/core")
def core():
    return manager.core_state()


@app.get("/e2")
def e2():
    return manager.e2_state()


@app.get("/intents")
def list_intents():
    return manager.list_intents()


@app.post("/intents")
def create_intent(payload: IntentPayload):
    return manager.create_intent(payload)


@app.delete("/intents/{intent_id}")
def delete_intent(intent_id: int):
    removed = manager.delete_intent(intent_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Intent not found")
    return {"status": "removed", "intent_id": intent_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ric_api:app", host="0.0.0.0", port=8000, reload=False)
