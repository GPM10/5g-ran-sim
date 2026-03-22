import logging
import os
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from controller import predictive_mobility_policy
from environment import NetworkEnvironment
from topology import (
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


class SimulationManager:
    def __init__(self, topology_path: Optional[str] = None):
        base_stations, users = self._load_topology(topology_path)
        self.env = NetworkEnvironment(base_stations, users)
        self._bs_id_to_index = {bs.bs_id: idx for idx, bs in enumerate(base_stations)}
        self.states, self.candidate_map = self.env.reset_for_rl()
        self.last_metrics: Optional[Dict[int, Dict[str, float]]] = None
        self.last_snapshot: Optional[Dict] = None
        self.last_timestamp = time.time()
        self.lock = Lock()
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
        mapping = {}
        for action in request.actions:
            if action.ue_id < 0 or action.ue_id >= len(self.env.users):
                logger.warning("Rejected action: invalid ue_id=%s", action.ue_id)
                raise HTTPException(status_code=400, detail=f"Invalid ue_id {action.ue_id}")
            if action.target_bs not in self._bs_id_to_index:
                logger.warning("Rejected action: invalid target_bs=%s", action.target_bs)
                raise HTTPException(
                    status_code=400, detail=f"Invalid target_bs {action.target_bs}"
                )
            mapping[action.ue_id] = action.target_bs
        with self.lock:
            action_vector = self._actions_from_mapping(mapping)
            self._step(action_vector, source="xapp")
            logger.info("Applied %d external actions", len(request.actions))
            return self.last_snapshot

    def get_history(self, tail: int = 50):
        with self.lock:
            history = {}
            for key, values in self.env.history.items():
                history[key] = values[-tail:]
            return history

    def _step(self, action_vector: Optional[List[int]] = None, source: Optional[str] = None):
        external = action_vector is not None
        if action_vector is None:
            action_vector = self._heuristic_actions()
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
            best_bs = predictive_mobility_policy(ue, self.env.base_stations)
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ric_api:app", host="0.0.0.0", port=8000, reload=False)
