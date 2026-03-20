import argparse
import os
import sys
import time
from typing import Dict, List

import httpx
import numpy as np
import torch

from rl_models import QNetwork


DEFAULT_API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")


def fetch_metrics(client: httpx.Client, base_url: str, advance: bool = True) -> Dict:
    response = client.get(f"{base_url}/metrics", params={"advance": advance})
    response.raise_for_status()
    return response.json()


def post_actions(client: httpx.Client, base_url: str, actions: List[Dict]) -> Dict:
    response = client.post(f"{base_url}/actions", json={"actions": actions})
    response.raise_for_status()
    return response.json()


def latency_aware_actions(snapshot: Dict, max_actions: int, load_threshold: float) -> List[Dict]:
    ue_stats = snapshot.get("ue_stats", [])
    if not ue_stats:
        return []
    state_matrix = snapshot.get("state_matrix", [])
    candidate_map = snapshot.get("candidate_map", [])
    if not state_matrix or not candidate_map:
        return []
    num_candidates = max(((len(state_matrix[0]) - 5) // 3), 1)
    ue_order = {ue["ue_id"]: idx for idx, ue in enumerate(ue_stats)}
    cell_loads = {cell["bs_id"]: cell.get("load", 0.0) for cell in snapshot.get("cell_stats", [])}

    ranked = sorted(
        ue_stats,
        key=lambda u: (
            u["latency_ms"] - u["latency_budget_ms"],
            u.get("queue_mbits", 0.0),
        ),
        reverse=True,
    )

    actions = []
    for ue in ranked:
        slack = ue["latency_ms"] - ue["latency_budget_ms"]
        if slack <= 0:
            break
        row_idx = ue_order[ue["ue_id"]]
        candidates = candidate_map[row_idx]
        if not candidates:
            continue
        state_vec = state_matrix[row_idx]
        best_score = None
        best_bs = None
        for c_idx, bs_id in enumerate(candidates):
            if c_idx >= num_candidates:
                break
            start = 5 + 3 * c_idx
            sig_norm, load_norm, cap_norm = state_vec[start : start + 3]
            network_load = cell_loads.get(bs_id, load_norm)
            if network_load >= load_threshold:
                continue
            score = 0.5 * (1.0 - load_norm) + 0.35 * sig_norm + 0.15 * cap_norm
            if best_score is None or score > best_score:
                best_score = score
                best_bs = bs_id
        if best_bs is None or best_bs == ue["serving_bs"]:
            continue
        actions.append({"ue_id": ue["ue_id"], "target_bs": best_bs})
        if len(actions) >= max_actions:
            break
    return actions


class DQNPolicy:
    def __init__(self, weights_path: str):
        self.weights_path = weights_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.state_dim = None
        self.num_candidates = None

    def plan(self, snapshot: Dict) -> List[Dict]:
        state_matrix = snapshot.get("state_matrix", [])
        candidate_map = snapshot.get("candidate_map", [])
        ue_stats = snapshot.get("ue_stats", [])
        if not state_matrix or not candidate_map:
            return []
        if self.state_dim is None:
            self.state_dim = len(state_matrix[0])
            self.num_candidates = max(((self.state_dim - 5) // 3), 1)
            self._load_model()
        states = torch.tensor(state_matrix, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_values = self.model(states).cpu().numpy()
        actions = []
        for row_idx, ue in enumerate(ue_stats):
            candidates = candidate_map[row_idx]
            if not candidates:
                continue
            limit = min(len(candidates), q_values.shape[1])
            best_idx = int(np.argmax(q_values[row_idx, :limit]))
            best_bs = candidates[min(best_idx, len(candidates) - 1)]
            actions.append({"ue_id": ue["ue_id"], "target_bs": best_bs})
        return actions

    def _load_model(self):
        self.model = QNetwork(self.state_dim, self.num_candidates).to(self.device)
        state_dict = torch.load(self.weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()


def format_summary(snapshot: Dict, prefix: str) -> str:
    aggregates = snapshot.get("aggregates", {})
    latency = aggregates.get("avg_latency_ms", 0.0)
    load = aggregates.get("avg_cell_load", 0.0)
    max_load = aggregates.get("max_cell_load", 0.0)
    throughput = aggregates.get("avg_throughput_mbps", 0.0)
    return (
        f"{prefix} | avg latency {latency:.1f} ms | avg load {load:.2f} | "
        f"max load {max_load:.2f} | avg thpt {throughput:.2f} Mbps"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Latency-aware xApp client")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="RIC metrics API base URL")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between control loops")
    parser.add_argument("--steps", type=int, default=0, help="Number of control iterations (0 = infinite)")
    parser.add_argument("--mode", choices=["heuristic", "dqn"], default="heuristic")
    parser.add_argument("--dqn-weights", type=str, default=None, help="Path to trained DQN weights")
    parser.add_argument("--max-actions", type=int, default=5, help="Max UE handovers per iteration in heuristic mode")
    parser.add_argument("--load-threshold", type=float, default=1.2, help="Skip cells with load beyond this value")
    parser.add_argument("--timeout", type=float, default=5.0, help="HTTP client timeout (seconds)")
    return parser.parse_args()


def main():
    args = parse_args()
    client = httpx.Client(timeout=args.timeout)
    policy = None
    if args.mode == "dqn":
        if not args.dqn_weights:
            print("--dqn-weights is required for DQN mode", file=sys.stderr)
            sys.exit(1)
        policy = DQNPolicy(args.dqn_weights)

    iteration = 0
    try:
        snapshot = fetch_metrics(client, args.api_base, advance=True)
        while True:
            iteration += 1
            if args.mode == "heuristic":
                actions = latency_aware_actions(
                    snapshot, max_actions=args.max_actions, load_threshold=args.load_threshold
                )
            else:
                actions = policy.plan(snapshot)

            if actions:
                snapshot = post_actions(client, args.api_base, actions)
                action_preview = ", ".join(
                    f"ue{a['ue_id']}?bs{a['target_bs']}" for a in actions[:5]
                )
                print(f"[step {iteration}] applied {len(actions)} actions: {action_preview}")
            else:
                snapshot = fetch_metrics(client, args.api_base, advance=True)
                print(f"[step {iteration}] no actions dispatched, baseline advanced")

            print("  " + format_summary(snapshot, prefix="metrics"))

            if args.steps and iteration >= args.steps:
                break
            time.sleep(max(args.interval, 0.1))
    except KeyboardInterrupt:
        print("Stopping xApp client...")
    finally:
        client.close()


if __name__ == "__main__":
    main()
