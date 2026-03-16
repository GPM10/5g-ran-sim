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
   python main.py  # add --no-plots on headless systems
   ```
3. Inspect the console summary plus four matplotlib plots (avg throughput, max load, avg latency, cumulative handovers) to compare baseline vs. load-aware policies.

## RL API (Phase 2)

`environment.py` now doubles as an RL-ready simulator:

- `reset_for_rl()` → resets mobility/load history and returns the stacked observation matrix plus each UE's candidate BS map.
- `build_state_matrix()` → can be called anytime to get the current observations without stepping the sim.
- `step_with_actions(actions)` → moves UEs, applies a discrete BS-selection action per UE, and returns `(next_states, rewards, info)` where `info["metrics"]` stores per-UE stats and `info["candidate_map"]` feeds the next action pass. Rewards balance throughput gains against overload, latency-budget violations, and handover costs (weights configurable via `DEFAULT_RL_CONFIG`).

**State layout:** `[demand_norm, latency_budget_norm, latency_slack_norm, last_handover_flag]` followed by, for each of the top-`k` candidate base stations, `[normalized signal strength, normalized previous load, normalized capacity]`. All values are clipped to `[0, 1]` (or `[-1, 1]` for latency slack) so a neural agent can ingest them directly.

**Action space:** one discrete choice per UE (index into its candidate list). Invalid actions are clipped to the closest candidate. Set `num_candidates` in `DEFAULT_RL_CONFIG` to control action dimensionality.

**Reward:** `w_t * (throughput / demand) - w_o * overload - w_l * latency_violation - w_h * handover_flag`, with tunable weights under `reward_weights`.

## Phase Roadmap

1. **(Done)** Better simulator – SINR instead of inverse-distance signal, logged handovers, heterogeneous demand, and per-UE latency estimates.
2. **(Done)** RL interface – normalized observations, candidate-based action space, and composite reward via `reset_for_rl`, `build_state_matrix`, and `step_with_actions`.
3. **(In progress)** DQN training – `train_dqn.py` provides a baseline replay-buffer + target-network implementation you can extend.

### Training the DQN prototype

```bash
python train_dqn.py --episodes 200 --steps 60 --output artifacts/dqn_weights.pt
```

- Logs per-episode reward, epsilon, and replay-buffer depth.
- Saves the learned weights to `artifacts/dqn_weights.pt` (create the folder if it does not exist).
- Tweak exploration or reward weights by editing `TrainConfig` / `DEFAULT_RL_CONFIG`.

### Evaluating a trained DQN

```bash
python main.py --dqn-weights artifacts/dqn_weights.pt --steps 60
```

This runs the baseline, load-aware heuristic, and the loaded DQN policy side by side, prints the KPI summary for each, and plots all histories (use `--no-plots` if running remotely).

## Notes

- Call the heuristic a *load-aware* or *AI-inspired* controller, reserving the AI label for the future RL version.
- Suggested résumé line: “Built a multi-cell 5G RAN simulator with mobile UEs, then developed a load-aware association controller that improved throughput, reduced congestion, and cut handovers versus strongest-signal baselines.”
