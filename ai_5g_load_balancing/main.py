import matplotlib.pyplot as plt
from models import BaseStation, UserEquipment
from environment import NetworkEnvironment
from controller import strongest_signal_policy, load_aware_policy


def build_network():
    base_stations = [
        BaseStation(0, 20, 20, capacity_mbps=40),
        BaseStation(1, 50, 80, capacity_mbps=40),
        BaseStation(2, 80, 30, capacity_mbps=40),
    ]

    users = [UserEquipment(i, x=(i * 7) % 100, y=(i * 13) % 100) for i in range(30)]
    return base_stations, users


def plot_results(hist1, hist2):
    plt.figure(figsize=(10, 4))
    plt.plot(hist1["avg_throughput"], label="Baseline Throughput")
    plt.plot(hist2["avg_throughput"], label="Load-Aware Throughput")
    plt.xlabel("Time step")
    plt.ylabel("Average Throughput (Mbps)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(hist1["max_load"], label="Baseline Max Load")
    plt.plot(hist2["max_load"], label="Load-Aware Max Load")
    plt.xlabel("Time step")
    plt.ylabel("Maximum Cell Load")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(hist1["avg_latency_ms"], label="Baseline Avg Latency")
    plt.plot(hist2["avg_latency_ms"], label="Load-Aware Avg Latency")
    plt.xlabel("Time step")
    plt.ylabel("Avg Latency (ms)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    baseline_cum = _cumulative(hist1["handover_count"])
    ai_cum = _cumulative(hist2["handover_count"])
    plt.plot(baseline_cum, label="Baseline Cum. Handovers")
    plt.plot(ai_cum, label="Load-Aware Cum. Handovers")
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
    print(f"{label} average SINR (dB): {hist['avg_sinr_db'][-1]:.2f}")
    print(f"{label} total handovers: {sum(hist['handover_count'])}")


def main():
    bs1, users1 = build_network()
    env1 = NetworkEnvironment(bs1, users1)
    baseline_hist = env1.run(strongest_signal_policy, steps=60)

    bs2, users2 = build_network()
    env2 = NetworkEnvironment(bs2, users2)
    ai_hist = env2.run(load_aware_policy, steps=60)

    print_summary("Baseline", baseline_hist)
    print_summary("Load-aware", ai_hist)

    plot_results(baseline_hist, ai_hist)


if __name__ == "__main__":
    main()
