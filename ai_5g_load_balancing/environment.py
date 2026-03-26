import time

import numpy as np
from .core import CoreNetwork
from .cu_du import DistributedUnit, build_cu_du_hierarchy, build_radio_units
from .interfaces import F1Interface, F1UPlanePacket, InterfaceLink
from .utils import (
    DEFAULT_TELEMETRY_COLLECTOR,
    estimate_latency_ms,
    signal_strength,
    sinr_linear,
    user_throughput_from_phy,
)

DEFAULT_RL_CONFIG = {
    "num_candidates": 2,
    "reward_weights": {
        "throughput": 1.0,
        "overload": 0.6,
        "latency": 0.6,
        "handover": 0.4,
        "queue": 0.2,
        "sinr": 0.3,
    },
    "max_demand_mbps": 8.0,
    "max_latency_ms": 200.0,
    "max_bs_capacity_mbps": 100.0,
    "max_ue_speed_m_s": 20.0,
    "timestep_s": 1.0,
    "scheduler_mode": "pf",  # pf | rr
    "sinr_target_db": 0.0,
    "demand_shape": {
        "season_period_steps": 200,
        "season_amplitude": 0.25,
        "spike_probability": 0.08,
        "spike_strength": [2.0, 6.0],
        "cluster_ratio": 0.3,
    },
}
SIGNAL_DBM_RANGE = (-120.0, -40.0)


class DemandShaper:
    def __init__(self, num_users, config=None):
        self.num_users = num_users
        cfg = config or {}
        self.season_period = max(int(cfg.get("season_period_steps", 120)), 1)
        self.season_amplitude = float(cfg.get("season_amplitude", 0.2))
        self.spike_probability = float(cfg.get("spike_probability", 0.05))
        strength = cfg.get("spike_strength", (2.0, 5.0))
        if isinstance(strength, (list, tuple)) and len(strength) == 2:
            self.spike_strength = (float(strength[0]), float(strength[1]))
        else:
            self.spike_strength = (2.0, 5.0)
        self.cluster_ratio = float(cfg.get("cluster_ratio", 0.25))
        self.current_step = 0
        self.last_event = None

    def reset(self):
        self.current_step = 0
        self.last_event = None

    def step(self):
        self.current_step += 1
        base = 1.0 + self.season_amplitude * np.sin(
            2 * np.pi * self.current_step / self.season_period
        )
        multipliers = np.full(self.num_users, base, dtype=np.float32)
        event = None
        if np.random.rand() < self.spike_probability:
            cluster_size = max(
                1, int(self.cluster_ratio * self.num_users)
            )
            affected = np.random.choice(
                self.num_users, cluster_size, replace=False
            )
            spike = np.random.uniform(*self.spike_strength)
            multipliers[affected] *= spike
            event = {
                "step": self.current_step,
                "affected_users": affected.tolist(),
                "multiplier": float(spike),
            }
        self.last_event = event
        return multipliers, event


class NetworkEnvironment:
    def __init__(
        self,
        base_stations,
        users,
        rl_config=None,
        telemetry=None,
        distributed_units=None,
        central_units=None,
        core_network=None,
    ):
        self.base_stations = base_stations
        self.users = users
        self.rl_config = DEFAULT_RL_CONFIG | (rl_config or {})
        self.telemetry = telemetry if telemetry is not None else DEFAULT_TELEMETRY_COLLECTOR
        self.timestep_s = self.rl_config.get("timestep_s", 1.0)
        self.scheduler_mode = self.rl_config.get("scheduler_mode", "pf")
        self.history = self._blank_history()
        self.prev_loads = np.zeros(len(self.base_stations), dtype=np.float32)
        self.last_latency = {u.ue_id: u.latency_budget_ms for u in self.users}
        self._last_candidate_map = None
        self._users_by_id = {u.ue_id: u for u in self.users}
        self.area_size = int(
            max(
                self.rl_config.get("area_size", 120),
                max(bs.x for bs in self.base_stations) + 10,
                max(bs.y for bs in self.base_stations) + 10,
            )
        )
        self.demand_shaper = DemandShaper(
            len(self.users), self.rl_config.get("demand_shape", {})
        )
        self.last_demand_event = None
        self._last_demand_factors = np.ones(len(self.users), dtype=np.float32)
        self.core_network = core_network if core_network is not None else CoreNetwork()
        if self.telemetry and self.core_network:
            self.core_network.attach_telemetry(self.telemetry)
        if distributed_units is None or central_units is None:
            self.distributed_units, self.central_units = build_cu_du_hierarchy(
                self.base_stations, num_cus=max(1, self.rl_config.get("cu_count", 2))
            )
        else:
            self.distributed_units = distributed_units
            self.central_units = central_units
        self.radio_units = build_radio_units(self.distributed_units)
        self.f1_interfaces, self._bs_to_du = self._build_f1_interfaces(self.distributed_units)
        self._last_f1_stats = {}
        self._last_ru_metrics = {}
        self._last_core_interface_stats = {}
        self._policy_context = {
            "bs_to_du": self._bs_to_du,
            "f1": {},
            "ru": {},
            "core": {"upf_load": 0.0, "interface": {}},
        }
        self._last_f1_stats = {du.du_id: {"control_packets": 0, "user_packets": 0, "control_queue": 0, "user_queue": 0} for du in self.distributed_units}

    def _blank_history(self):
        return {
            "avg_load": [],
            "avg_throughput": [],
            "max_load": [],
            "avg_latency_ms": [],
            "avg_queue_mbits": [],
            "handover_count": [],
            "avg_sinr_db": [],
        }

    def reset_history(self):
        self.history = self._blank_history()

    def reset_bs(self):
        for bs in self.base_stations:
            bs.reset()
        for f1 in self.f1_interfaces.values():
            f1.control._queue.clear()
            f1.user._queue.clear()
        for du_id in self._last_f1_stats:
            self._last_f1_stats[du_id] = {"control_packets": 0, "user_packets": 0, "control_queue": 0, "user_queue": 0}
        for bs_id in self._last_ru_metrics:
            self._last_ru_metrics[bs_id] = {"load": 0.0, "temperature_c": 40.0}
        self._update_policy_context()

    @property
    def policy_context(self):
        return self._policy_context

    def _build_f1_interfaces(self, distributed_units: List[DistributedUnit]):
        f1_map = {}
        bs_to_du = {}
        for du in distributed_units:
            control_link = InterfaceLink(
                f"F1C-{du.du_id}",
                latency_ms=du.fronthaul_latency_ms * 1.2,
                jitter_ms=0.2,
                drop_rate=0.005,
            )
            user_link = InterfaceLink(
                f"F1U-{du.du_id}",
                latency_ms=du.fronthaul_latency_ms,
                jitter_ms=0.1,
                drop_rate=0.002,
                bandwidth_mbps=du.fronthaul_capacity_gbps * 1000.0,
            )
            f1_map[du.du_id] = F1Interface(
                name=du.du_id, control_link=control_link, user_link=user_link
            )
            for bs in du.base_stations:
                bs_to_du[bs.bs_id] = du.du_id
        return f1_map, bs_to_du

    # ------------------------------------------------------------------
    # Classic heuristic-driven step/run
    # ------------------------------------------------------------------
    def step(self, association_policy):
        handover_events = {}
        demand_factors = self._next_demand_factors()
        self._ensure_core_sessions()
        for idx, u in enumerate(self.users):
            u.move(
                area_size=self.area_size,
                slot_duration_s=self.timestep_s,
                demand_factor=demand_factors[idx],
            )

        self.reset_bs()

        for u in self.users:
            previous_bs = u.serving_bs
            selected_bs = association_policy(u, self.base_stations, self.policy_context)
            if previous_bs is not None and previous_bs != selected_bs.bs_id:
                handover_events[u.ue_id] = 1
            else:
                handover_events[u.ue_id] = 0
            u.previous_bs = previous_bs
            u.serving_bs = selected_bs.bs_id
            selected_bs.add_user(u)
            self._enqueue_f1_control(u, previous_bs, selected_bs.bs_id)

        self._finalize_step(handover_events, update_history=True)

    def run(self, association_policy, steps=50):
        self.reset_history()
        for _ in range(steps):
            self.step(association_policy)
        return self.history

    # ------------------------------------------------------------------
    # RL-friendly API
    # ------------------------------------------------------------------
    def reset_for_rl(self):
        self.reset_history()
        self.reset_bs()
        self.prev_loads = np.zeros(len(self.base_stations), dtype=np.float32)
        self.last_latency = {
            u.ue_id: u.latency_budget_ms for u in self.users
        }
        self.demand_shaper.reset()
        for u in self.users:
            u.serving_bs = None
            u.previous_bs = None
            u.backlog_mbits = 0.0
        states, candidate_map = self.build_state_matrix()
        self._last_candidate_map = candidate_map
        return states, candidate_map

    def step_with_actions(self, actions, candidate_map=None, update_history=False):
        if candidate_map is None:
            if self._last_candidate_map is None:
                raise ValueError("Call build_state_matrix() before taking actions")
            candidate_map = self._last_candidate_map

        actions = np.asarray(actions, dtype=np.int32)
        if actions.shape[0] != len(self.users):
            raise ValueError("Action vector length must equal number of UEs")

        demand_factors = self._next_demand_factors()
        for idx, u in enumerate(self.users):
            u.move(
                area_size=self.area_size,
                slot_duration_s=self.timestep_s,
                demand_factor=demand_factors[idx],
            )
        self.reset_bs()
        self._ensure_core_sessions()

        handover_events = {}
        for idx, u in enumerate(self.users):
            candidates = candidate_map[idx]
            if not candidates:
                candidates = self._get_candidate_bs(u)
            action = int(actions[idx]) if candidates else 0
            action = int(np.clip(action, 0, max(len(candidates) - 1, 0)))
            target_bs_id = candidates[action]
            selected_bs = self.base_stations[target_bs_id]
            previous_bs = u.serving_bs
            handover_events[u.ue_id] = (
                1 if previous_bs is not None and previous_bs != target_bs_id else 0
            )
            u.previous_bs = previous_bs
            u.serving_bs = target_bs_id
            selected_bs.add_user(u)
            self._enqueue_f1_control(u, previous_bs, target_bs_id)

        per_user_metrics = self._finalize_step(
            handover_events, update_history=update_history
        )
        rewards = self._compute_rewards(per_user_metrics, handover_events)
        next_states, next_candidate_map = self.build_state_matrix()
        self._last_candidate_map = next_candidate_map
        info = {
            "metrics": per_user_metrics,
            "candidate_map": next_candidate_map,
        }
        return next_states, rewards, info

    def build_state_matrix(self):
        states = []
        candidate_map = []
        for u in self.users:
            candidates = self._get_candidate_bs(u)
            state_vec = self._build_state_vector(u, candidates)
            candidate_map.append(candidates)
            states.append(state_vec)
        return np.vstack(states), candidate_map

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _next_demand_factors(self):
        multipliers, event = self.demand_shaper.step()
        self.last_demand_event = event
        self._last_demand_factors = multipliers
        return multipliers

    def _schedule_users(self, bs):
        users = bs.connected_users
        if not users:
            bs.update_load(0.0)
            return {}
        if self.scheduler_mode == "rr":
            weights = np.ones(len(users))
        else:  # proportional fairness style
            weights = np.array(
                [1.0 / max(u.avg_throughput_mbps, 0.1) for u in users]
            )
        weights = weights / max(weights.sum(), 1e-9)

        def _allocate(total_rbs):
            total_rbs = max(total_rbs, 1)
            raw_alloc = weights * total_rbs
            rb_alloc = np.floor(raw_alloc).astype(int)
            remaining = total_rbs - rb_alloc.sum()
            if remaining > 0:
                fractional = raw_alloc - rb_alloc
                order = np.argsort(-fractional)
                for idx in order[:remaining]:
                    rb_alloc[idx] += 1
            return {u.ue_id: rb_alloc[i] for i, u in enumerate(users)}

        carrier_allocations = {}
        for carrier in bs.iter_carriers():
            carrier_allocations[carrier.name] = _allocate(carrier.resource_blocks)

        total_request_rate = sum(
            u.backlog_mbits / max(self.timestep_s, 1e-6) for u in users
        )
        total_capacity = sum(c.capacity_mbps for c in bs.iter_carriers())
        load_ratio = min(total_request_rate / max(total_capacity, 1e-3), 3.0)
        bs.update_load(load_ratio)
        return carrier_allocations

    def _finalize_step(self, handover_events, update_history):
        throughputs = []
        loads = []
        latencies = []
        queue_latencies = []
        sinrs_db = []
        per_user_metrics = {}
        bs_throughput = {}

        for bs_idx, bs in enumerate(self.base_stations):
            carrier_allocations = self._schedule_users(bs)
            loads.append(bs.load)
            for u in bs.connected_users:
                backlog_before = u.backlog_mbits
                user_throughput = 0.0
                sinr_samples = []
                carrier_rates = []
                for carrier in bs.iter_carriers():
                    alloc_map = carrier_allocations.get(carrier.name, {})
                    alloc_rbs = alloc_map.get(u.ue_id, 0)
                    if alloc_rbs <= 0:
                        continue
                    sinr_val = sinr_linear(
                        u, bs, self.base_stations, carrier, fading_config=self.rl_config.get("fading")
                    )
                    thpt, phy_rate, cqi = user_throughput_from_phy(
                        u,
                        carrier.bandwidth_mhz,
                        carrier.resource_blocks,
                        sinr_val,
                        alloc_rbs,
                        timestep_s=self.timestep_s,
                    )
                    sinr_samples.append(10 * np.log10(sinr_val + 1e-9))
                    carrier_rates.append(
                        {
                            "carrier": carrier.name,
                            "throughput_mbps": thpt,
                            "allocated_rbs": alloc_rbs,
                            "phy_rate_mbps": phy_rate,
                        }
                    )
                    user_throughput += thpt
                u.update_avg_throughput(user_throughput)
                latency = estimate_latency_ms(
                    u, user_throughput, backlog_before, timestep_s=self.timestep_s
                )
                throughputs.append(user_throughput)
                latencies.append(latency)
                queue_latencies.append(backlog_before)
                sinrs_db.append(
                    float(np.mean(sinr_samples)) if sinr_samples else -np.inf
                )
                bs_throughput[bs.bs_id] = bs_throughput.get(bs.bs_id, 0.0) + user_throughput
                per_user_metrics[u.ue_id] = {
                    "throughput": user_throughput,
                    "latency": latency,
                    "queue_mbits": backlog_before,
                    "cell_load": bs.load,
                    "sinr_db": float(np.mean(sinr_samples)) if sinr_samples else -np.inf,
                    "carrier_rates": carrier_rates,
                }
                self.last_latency[u.ue_id] = latency
                self._send_f1_user_packet(bs.bs_id, u, user_throughput, latency)

        if loads:
            self.prev_loads = np.array(loads, dtype=np.float32)
        else:
            self.prev_loads = np.zeros(len(self.base_stations), dtype=np.float32)

        if update_history:
            self.history["avg_load"].append(float(np.mean(loads)) if loads else 0.0)
            self.history["max_load"].append(float(np.max(loads)) if loads else 0.0)
            self.history["avg_throughput"].append(
                float(np.mean(throughputs)) if throughputs else 0.0
            )
            self.history["avg_latency_ms"].append(
                float(np.mean(latencies)) if latencies else 0.0
            )
            self.history["avg_queue_mbits"].append(
                float(np.mean(queue_latencies)) if queue_latencies else 0.0
            )
            total_handovers = sum(handover_events.values())
            self.history["handover_count"].append(int(total_handovers))
            self.history["avg_sinr_db"].append(
                float(np.mean(sinrs_db)) if sinrs_db else -np.inf
            )

        if self.telemetry:
            self._emit_kpis(
                loads=loads,
                throughputs=throughputs,
                latencies=latencies,
                queue_latencies=queue_latencies,
                handover_events=handover_events,
                per_user_metrics=per_user_metrics,
                bs_throughput=bs_throughput,
            )

        return per_user_metrics

    def _get_candidate_bs(self, ue):
        num_candidates = int(self.rl_config.get("num_candidates", 3))
        scored = []
        for idx, bs in enumerate(self.base_stations):
            scored.append((signal_strength(ue, bs), idx))
        scored.sort(key=lambda item: item[0], reverse=True)
        top = [idx for _, idx in scored[:num_candidates]]
        return top if top else [0]

    def _build_state_vector(self, ue, candidate_ids):
        cfg = self.rl_config
        max_demand = cfg.get("max_demand_mbps", 8.0)
        max_latency = cfg.get("max_latency_ms", 200.0)
        max_capacity = cfg.get("max_bs_capacity_mbps", 100.0)
        backlog_rate = ue.backlog_mbits / max(self.timestep_s, 1e-6)

        demand_norm = np.clip(ue.demand / max_demand, 0.0, 1.0)
        backlog_norm = np.clip(backlog_rate / max_demand, 0.0, 1.0)
        latency_norm = np.clip(ue.latency_budget_ms / max_latency, 0.0, 1.0)
        last_latency = self.last_latency.get(ue.ue_id, ue.latency_budget_ms)
        latency_slack = ue.latency_budget_ms - last_latency
        slack_norm = np.clip(
            latency_slack / max(ue.latency_budget_ms, 1e-6), -1.0, 1.0
        )
        handover_flag = (
            1.0 if ue.previous_bs is not None and ue.previous_bs != ue.serving_bs else 0.0
        )
        max_speed = cfg.get("max_ue_speed_m_s", 20.0)
        speed_norm = np.clip(
            getattr(ue, "velocity_m_s", 0.0) / max(max_speed, 1e-3), 0.0, 1.0
        )
        heading_norm = (getattr(ue, "heading_deg", 0.0) % 360.0) / 360.0
        state = [
            demand_norm,
            backlog_norm,
            latency_norm,
            slack_norm,
            handover_flag,
            speed_norm,
            heading_norm,
        ]

        for bs_id in candidate_ids:
            bs = self.base_stations[bs_id]
            sig_dbm = signal_strength(ue, bs)
            sig_norm = np.clip(
                (sig_dbm - SIGNAL_DBM_RANGE[0])
                / (SIGNAL_DBM_RANGE[1] - SIGNAL_DBM_RANGE[0]),
                0.0,
                1.0,
            )
            load_norm = np.clip(self.prev_loads[bs_id], 0.0, 2.0) / 2.0
            capacity_norm = np.clip(bs.capacity_mbps / max_capacity, 0.0, 1.0)
            state.extend([sig_norm, load_norm, capacity_norm])

        missing = int(cfg.get("num_candidates", 3)) - len(candidate_ids)
        if missing > 0:
            state.extend([0.0, 0.0, 0.0] * missing)

        return np.array(state, dtype=np.float32)

    def _compute_rewards(self, per_user_metrics, handover_events):
        weights = self.rl_config.get("reward_weights", {})
        w_t = weights.get("throughput", 1.0)
        w_o = weights.get("overload", 0.5)
        w_l = weights.get("latency", 0.5)
        w_h = weights.get("handover", 0.1)
        w_q = weights.get("queue", 0.0)
        w_s = weights.get("sinr", 0.0)
        sinr_target = self.rl_config.get("sinr_target_db", 0.0)

        rewards = []
        for u in self.users:
            metrics = per_user_metrics.get(
                u.ue_id,
                {
                    "throughput": 0.0,
                    "latency": u.latency_budget_ms,
                    "cell_load": 0.0,
                    "queue_mbits": 0.0,
                    "sinr_db": 0.0,
                },
            )
            throughput_ratio = metrics["throughput"] / max(u.demand, 1e-6)
            overload_pen = max(metrics["cell_load"] - 1.0, 0.0)
            latency_pen = max(
                metrics["latency"] - u.latency_budget_ms, 0.0
            ) / max(u.latency_budget_ms, 1e-6)
            queue_pen = metrics.get("queue_mbits", 0.0) / 20.0
            sinr_deficit = max(sinr_target - metrics.get("sinr_db", sinr_target), 0.0)
            sinr_pen = sinr_deficit / max(abs(sinr_target) + 1e-3, 1.0)
            handover_pen = handover_events.get(u.ue_id, 0)
            reward = (
                w_t * throughput_ratio
                - w_o * overload_pen
                - w_l * latency_pen
                - w_q * queue_pen
                - w_s * sinr_pen
                - w_h * handover_pen
            )
            rewards.append(float(reward))
        return np.array(rewards, dtype=np.float32)

    # ------------------------------------------------------------------
    # Telemetry helpers
    # ------------------------------------------------------------------
    def set_telemetry_exporters(self, exporters):
        if self.telemetry:
            self.telemetry.set_exporters(exporters)

    def _emit_kpis(
        self,
        loads,
        throughputs,
        latencies,
        queue_latencies,
        handover_events,
        per_user_metrics,
        bs_throughput,
    ):
        timestamp = time.time()
        self._service_f1_interfaces()
        total_bs = len(self.base_stations)
        for bs in self.base_stations:
            self.telemetry.emit(
                "radio.cell_load_ratio",
                float(bs.load),
                labels={"bs_id": bs.bs_id, "tier": bs.tier},
                timestamp=timestamp,
            )
        if self.radio_units:
            for ru in self.radio_units:
                util = ru.update_metrics()
                labels = ru.telemetry_labels()
                self.telemetry.emit(
                    "radio.ru_load_ratio",
                    float(util),
                    labels=labels,
                    timestamp=timestamp,
                )
                self.telemetry.emit(
                    "radio.ru_temperature_c",
                    float(ru.temperature_c),
                    labels=labels,
                    timestamp=timestamp,
                )
                self._last_ru_metrics[ru.base_station.bs_id] = {
                    "load": util,
                    "temperature_c": ru.temperature_c,
                }

        avg_throughput = float(np.mean(throughputs)) if throughputs else 0.0
        avg_latency = float(np.mean(latencies)) if latencies else 0.0
        avg_queue = float(np.mean(queue_latencies)) if queue_latencies else 0.0
        overloaded_cells = sum(1 for load in loads if load >= 1.0)

        self.telemetry.emit(
            "network.avg_throughput_mbps",
            avg_throughput,
            labels={"scope": "network"},
            timestamp=timestamp,
        )
        self.telemetry.emit(
            "network.avg_latency_ms",
            avg_latency,
            labels={"scope": "network"},
            timestamp=timestamp,
        )
        self.telemetry.emit(
            "network.queue_backlog_mbits",
            avg_queue,
            labels={"scope": "network"},
            timestamp=timestamp,
        )
        self.telemetry.emit(
            "network.overloaded_cells",
            float(overloaded_cells),
            labels={"scope": "network", "total_bs": total_bs},
            timestamp=timestamp,
        )

        attempts = int(sum(handover_events.values()))
        successes = self._estimate_handover_success(handover_events, per_user_metrics)
        failures = max(attempts - successes, 0)
        success_ratio = successes / attempts if attempts > 0 else 1.0

        self.telemetry.emit(
            "mobility.handover_attempts_total",
            float(attempts),
            labels={"scope": "network"},
            timestamp=timestamp,
        )
        self.telemetry.emit(
            "mobility.handover_failure_total",
            float(failures),
            labels={"scope": "network"},
            timestamp=timestamp,
        )
        self.telemetry.emit(
            "mobility.handover_success_ratio",
            float(success_ratio),
            labels={"scope": "network"},
            timestamp=timestamp,
        )

        if self.distributed_units:
            for du in self.distributed_units:
                du.update_load(bs_throughput)
                du_labels = du.telemetry_labels()
                self.telemetry.emit(
                    "core.du_processing_utilization",
                    float(du.utilization),
                    labels=du_labels,
                    timestamp=timestamp,
                )
                self.telemetry.emit(
                    "core.du_fronthaul_utilization",
                    float(du.fronthaul_utilization),
                    labels=du_labels,
                    timestamp=timestamp,
                )
                stats = self._last_f1_stats.get(du.du_id, {})
                f1 = self.f1_interfaces.get(du.du_id)
                if f1:
                    self.telemetry.emit(
                        "interface.f1_control_queue",
                        float(stats.get("control_queue", f1.control.queue_depth())),
                        labels={"du_id": du.du_id, "cu_id": du.cu_id},
                        timestamp=timestamp,
                    )
                    self.telemetry.emit(
                        "interface.f1_user_queue",
                        float(stats.get("user_queue", f1.user.queue_depth())),
                        labels={"du_id": du.du_id, "cu_id": du.cu_id},
                        timestamp=timestamp,
                    )
                    self.telemetry.emit(
                        "interface.f1_control_packets",
                        float(stats.get("control_packets", 0)),
                        labels={"du_id": du.du_id},
                        timestamp=timestamp,
                    )
                    self.telemetry.emit(
                        "interface.f1_user_packets",
                        float(stats.get("user_packets", 0)),
                        labels={"du_id": du.du_id},
                        timestamp=timestamp,
                    )

        if self.central_units:
            for cu in self.central_units:
                util = cu.update_utilization()
                self.telemetry.emit(
                    "core.cu_cpu_utilization",
                    float(util),
                    labels=cu.telemetry_labels(),
                    timestamp=timestamp,
                )

        if self.core_network:
            per_user_throughput = {
                ue_id: metrics.get("throughput", 0.0)
                for ue_id, metrics in per_user_metrics.items()
            }
            self.core_network.update_traffic(per_user_throughput)
            self._last_core_interface_stats = self.core_network.service_interfaces() or {}
            self.core_network.emit_metrics(timestamp)
        self._update_policy_context()

    def _estimate_handover_success(self, handover_events, per_user_metrics):
        successes = 0
        for ue_id, occurred in handover_events.items():
            if not occurred:
                continue
            metrics = per_user_metrics.get(ue_id)
            ue = self._users_by_id.get(ue_id)
            if not metrics or ue is None:
                continue
            load_ok = metrics.get("cell_load", 0.0) <= 1.2
            latency_ok = metrics.get("latency", np.inf) <= ue.latency_budget_ms * 1.2
            if load_ok and latency_ok:
                successes += 1
        return successes

    def _ensure_core_sessions(self):
        if not self.core_network:
            return
        for u in self.users:
            slice_id = getattr(u, "traffic_profile", "embb")
            qos_profile = "latency" if u.latency_budget_ms <= 20 else "default"
            self.core_network.ensure_session(u.ue_id, slice_id, qos_profile)

    def _enqueue_f1_control(self, ue, previous_bs_id, new_bs_id):
        if previous_bs_id == new_bs_id:
            return
        du_id = self._bs_to_du.get(new_bs_id)
        if not du_id:
            return
        interface = self.f1_interfaces.get(du_id)
        if not interface:
            return
        payload = {
            "ue_id": ue.ue_id,
            "from_bs": previous_bs_id,
            "to_bs": new_bs_id,
            "timestamp": time.time(),
        }
        interface.send_control(payload)

    def _send_f1_user_packet(self, bs_id, ue, throughput_mbps, latency_ms):
        du_id = self._bs_to_du.get(bs_id)
        if not du_id:
            return
        interface = self.f1_interfaces.get(du_id)
        if not interface:
            return
        packet = F1UPlanePacket(
            ue_id=ue.ue_id,
            throughput_mbps=throughput_mbps,
            latency_ms=latency_ms,
        )
        interface.send_user_plane(packet)

    def _service_f1_interfaces(self):
        if not self.f1_interfaces:
            return
        for du_id, interface in self.f1_interfaces.items():
            deliveries = interface.poll()
            stats = {
                "control_packets": len(deliveries.get("control", [])),
                "user_packets": len(deliveries.get("user", [])),
                "control_queue": interface.control.queue_depth(),
                "user_queue": interface.user.queue_depth(),
            }
            self._last_f1_stats[du_id] = stats
        self._update_policy_context()

    def _update_policy_context(self):
        core_context = {
            "upf_load": self.core_network.upf.load_ratio if self.core_network else 0.0,
            "interface": self._last_core_interface_stats or {},
        }
        self._policy_context = {
            "bs_to_du": self._bs_to_du,
            "f1": dict(self._last_f1_stats),
            "ru": dict(self._last_ru_metrics),
            "core": core_context,
        }
