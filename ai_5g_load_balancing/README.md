# AI-Driven 5G Load Balancing Demo

This lightweight simulator compares a strongest-signal baseline against a load-aware "AI-inspired" heuristic for associating mobile UEs to 5G base stations. It now uses SINR-based radio modeling, heterogeneous traffic classes, latency estimation, and handover logging so you can tell a richer story before moving to RL.

## Project Structure

```
ai_5g_load_balancing/
├── main.py              # runs baseline vs heuristic comparison + plotting
├── environment.py       # mobility, association, SINR/latency bookkeeping
├── models.py            # base-station and heterogeneous UE data classes
├── controller.py        # strongest-signal and load-aware policies
├── utils.py             # signal math (distance, SINR, latency helpers)
├── train_dqn.py         # placeholder for future RL controller
├── requirements.txt     # python deps
└── README.md
```

## Quick Start

1. Create a virtual environment (recommended) and install deps:
   ```bash
   cd ai_5g_load_balancing
   python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Run the comparison demo:
   ```bash
   python main.py
   ```
3. Inspect the console summary plus four matplotlib plots (avg throughput, max load, avg latency, cumulative handovers) to compare baseline vs. load-aware policies.

## Phase Roadmap

1. **(Done)** Better simulator – SINR instead of inverse-distance signal, logged handovers, heterogeneous demand, and per-UE latency estimates.
2. **State/action/reward design** – expose per-user signal snapshots, BS loads, demand and define a reward with throughput, overload, and handover penalties.
3. **RL controller** – implement DQN in `train_dqn.py` (replay buffer, epsilon-greedy, Q-network, target net) and swap it into `main.py` for evaluation.

## Notes

- Call the heuristic a *load-aware* or *AI-inspired* controller, reserving the AI label for the future RL version.
- Suggested résumé line: “Built a multi-cell 5G RAN simulator with mobile UEs, then developed a load-aware association controller that improved throughput, reduced congestion, and cut handovers versus strongest-signal baselines.”
