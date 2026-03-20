import numpy as np
from utils import (
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
    "timestep_s": 1.0,
    "scheduler_mode": "pf",  # pf | rr
    "sinr_target_db": 0.0,
}
SIGNAL_DBM_RANGE = (-120.0, -40.0)


class NetworkEnvironment:
    def __init__(self, base_stations, users, rl_config=None):
        self.base_stations = base_stations
        self.users = users
        self.rl_config = DEFAULT_RL_CONFIG | (rl_config or {})
        self.timestep_s = self.rl_config.get("timestep_s", 1.0)
        self.scheduler_mode = self.rl_config.get("scheduler_mode", "pf")
        self.history = self._blank_history()
        self.prev_loads = np.zeros(len(self.base_stations), dtype=np.float32)
        self.last_latency = {u.ue_id: u.latency_budget_ms for u in self.users}
        self._last_candidate_map = None

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

    # ------------------------------------------------------------------
    # Classic heuristic-driven step/run
    # ------------------------------------------------------------------
    def step(self, association_policy):
        handover_events = {}
        for u in self.users:
            u.move(slot_duration_s=self.timestep_s)

        self.reset_bs()

        for u in self.users:
            previous_bs = u.serving_bs
            selected_bs = association_policy(u, self.base_stations)
            if previous_bs is not None and previous_bs != selected_bs.bs_id:
                handover_events[u.ue_id] = 1
            else:
                handover_events[u.ue_id] = 0
            u.previous_bs = previous_bs
            u.serving_bs = selected_bs.bs_id
            selected_bs.add_user(u)

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

        for u in self.users:
            u.move(slot_duration_s=self.timestep_s)
        self.reset_bs()

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
    def _schedule_users(self, bs):
        users = bs.connected_users
        if not users:
            bs.update_load(0.0)
            return {}
        total_rbs = max(bs.resource_blocks, 1)
        if self.scheduler_mode == "rr":
            weights = np.ones(len(users))
        else:  # proportional fairness style
            weights = np.array(
                [1.0 / max(u.avg_throughput_mbps, 0.1) for u in users]
            )
        weights = weights / weights.sum()
        raw_alloc = weights * total_rbs
        rb_alloc = np.floor(raw_alloc).astype(int)
        remaining = total_rbs - rb_alloc.sum()
        if remaining > 0:
            fractional = raw_alloc - rb_alloc
            order = np.argsort(-fractional)
            for idx in order[:remaining]:
                rb_alloc[idx] += 1
        allocations = {u.ue_id: rb_alloc[i] for i, u in enumerate(users)}
        total_request_rate = sum(
            u.backlog_mbits / max(self.timestep_s, 1e-6) for u in users
        )
        load_ratio = min(total_request_rate / bs.capacity_mbps, 3.0)
        bs.update_load(load_ratio)
        return allocations

    def _finalize_step(self, handover_events, update_history):
        throughputs = []
        loads = []
        latencies = []
        queue_latencies = []
        sinrs_db = []
        per_user_metrics = {}

        for bs_idx, bs in enumerate(self.base_stations):
            allocations = self._schedule_users(bs)
            loads.append(bs.load)
            for u in bs.connected_users:
                alloc_rbs = allocations.get(u.ue_id, max(bs.resource_blocks // len(bs.connected_users), 1))
                sinr_val = sinr_linear(u, bs, self.base_stations)
                sinr_db = 10 * np.log10(sinr_val + 1e-9)
                backlog_before = u.backlog_mbits
                thpt, phy_rate, cqi = user_throughput_from_phy(
                    u, bs, sinr_val, alloc_rbs, timestep_s=self.timestep_s
                )
                u.update_avg_throughput(thpt)
                latency = estimate_latency_ms(
                    u, thpt, backlog_before, timestep_s=self.timestep_s
                )
                throughputs.append(thpt)
                latencies.append(latency)
                queue_latencies.append(backlog_before)
                sinrs_db.append(sinr_db)
                per_user_metrics[u.ue_id] = {
                    "throughput": thpt,
                    "latency": latency,
                    "queue_mbits": backlog_before,
                    "cell_load": bs.load,
                    "sinr_db": sinr_db,
                    "radio_rate": phy_rate,
                    "cqi": cqi,
                    "allocated_rbs": alloc_rbs,
                }
                self.last_latency[u.ue_id] = latency

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
        state = [demand_norm, backlog_norm, latency_norm, slack_norm, handover_flag]

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
