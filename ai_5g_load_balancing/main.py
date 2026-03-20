import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from controller import load_aware_policy, strongest_signal_policy
from environment import NetworkEnvironment
from rl_models import QNetwork
from topology import build_network_from_spec, build_reference_network, load_topology_spec


class DQNInferenceAgent:
    def __init__(self, state_dim, action_dim, weights_path, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = QNetwork(state_dim, action_dim).to(self.device)
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def act(self, states, candidate_map):
        state_tensor = torch.from_numpy(states).float().to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().numpy()
        actions = []
        for i, candidates in enumerate(candidate_map):
            if len(candidates) == 0:
                actions.append(0)
            else:
                actions.append(int(np.argmax(q_values[i, : len(candidates)])))
        return actions


def build_network(topology_path=None):
    if topology_path:
        spec = load_topology_spec(topology_path)
        return build_network_from_spec(spec)
    return build_reference_network()


def run_dqn_episode(weights_path, steps, topology_path=None):
    bs, users = build_network(topology_path)
    env = NetworkEnvironment(bs, users)
    states, candidate_map = env.reset_for_rl()
    agent = DQNInferenceAgent(
        state_dim=states.shape[1],
        action_dim=env.rl_config.get("num_candidates", 3),
        weights_path=weights_path,
    )

    for _ in range(steps):
        actions = agent.act(states, candidate_map)
        states, _, info = env.step_with_actions(
            actions, candidate_map=candidate_map, update_history=True
        )
        candidate_map = info["candidate_map"]
    return env.history


def plot_results(histories):
    plt.figure(figsize=(10, 4))
    for label, hist in histories.items():
        plt.plot(hist["avg_throughput"], label=f"{label} Throughput")
    plt.xlabel("Time step")
    plt.ylabel("Average Throughput (Mbps)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    for label, hist in histories.items():
        plt.plot(hist["avg_queue_mbits"], label=f"{label} Avg Queue (Mbits)")
    plt.xlabel("Time step")
    plt.ylabel("Avg Queue Backlog (Mbits)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    for label, hist in histories.items():
        plt.plot(hist["max_load"], label=f"{label} Max Load")
    plt.xlabel("Time step")
    plt.ylabel("Maximum Cell Load")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    for label, hist in histories.items():
        plt.plot(hist["avg_latency_ms"], label=f"{label} Avg Latency")
    plt.xlabel("Time step")
    plt.ylabel("Avg Latency (ms)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    for label, hist in histories.items():
        cum = _cumulative(hist["handover_count"])
        plt.plot(cum, label=f"{label} Cum. Handovers")
    plt.xlabel("Time step")
    plt.ylabel("Cumulative Handovers")
    plt.legend()
    plt.tight_layout()
    plt.show()


def _cumulative(values):
    total = 0
    accum = []
    for val in values:
        total += val
        accum.append(total)
    return accum


def print_summary(label, hist):
    print(f"{label} final avg throughput (Mbps): {hist['avg_throughput'][-1]:.2f}")
    print(f"{label} final max load: {hist['max_load'][-1]:.2f}")
    print(f"{label} final avg latency (ms): {hist['avg_latency_ms'][-1]:.2f}")
    print(f"{label} avg queue backlog (Mbits): {hist['avg_queue_mbits'][-1]:.2f}")
    print(f"{label} average SINR (dB): {hist['avg_sinr_db'][-1]:.2f}")
    print(f"{label} total handovers: {sum(hist['handover_count'])}")


def parse_args():
    parser = argparse.ArgumentParser(description="5G load-balancing demo")
    parser.add_argument("--steps", type=int, default=60, help="Simulation steps")
    parser.add_argument(
        "--dqn-weights",
        type=str,
        default=None,
        help="Path to trained DQN weights for evaluation",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib plotting (useful for headless runs)",
    )
    parser.add_argument(
        "--topology",
        type=str,
        default=None,
        help="Path to a JSON topology spec (defaults to built-in macro/micro)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    steps = args.steps

    bs1, users1 = build_network(args.topology)
    env1 = NetworkEnvironment(bs1, users1)
    baseline_hist = env1.run(strongest_signal_policy, steps=steps)

    bs2, users2 = build_network(args.topology)
    env2 = NetworkEnvironment(bs2, users2)
    ai_hist = env2.run(load_aware_policy, steps=steps)

    histories = {"Baseline": baseline_hist, "Load-aware": ai_hist}

    if args.dqn_weights:
        dqn_hist = run_dqn_episode(args.dqn_weights, steps=steps, topology_path=args.topology)
        histories["DQN"] = dqn_hist

    for label, hist in histories.items():
        print_summary(label, hist)

    if not args.no_plots:
        plot_results(histories)


if __name__ == "__main__":
    main()
