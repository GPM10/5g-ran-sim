# AI-Driven 5G Load Balancing Demo

This lightweight simulator compares a strongest-signal baseline against a load-aware "AI-inspired" heuristic for associating mobile UEs to 5G base stations. It now includes multi-tier (macro + micro) cells, LOS/NLOS-aware SINR modeling with stochastic fading, heterogeneous traffic classes with backlog-based latency, proportional-fair / round-robin schedulers, and handover logging so you can tell a richer story before moving to RL.

## Project Structure

```
ai_5g_load_balancing/
├── main.py              # runs baseline vs heuristic comparison + plotting
├── environment.py       # mobility, association, queues, scheduling, SINR bookkeeping
├── models.py            # base-station tiers + heterogeneous UE traffic profiles
├── controller.py        # strongest-signal and load-aware policies
├── topology.py          # reusable builders + JSON-driven topology loader
├── utils.py             # signal math (distance, SINR, latency helpers)
├── train_dqn.py         # baseline DQN training loop + replay buffer
├── topologies/          # sample JSON specs (e.g., hotspot_dense.json)
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
3. Inspect the console summary plus five matplotlib plots (avg throughput, max load, avg latency, average queue backlog, cumulative handovers) to compare baseline vs. load-aware policies.

### Simulator realism knobs

- **Multi-tier deployment** – macros transmit at higher power/bandwidth, micros densify hotspots. LOS/NLOS probability and path-loss parameters are tier-aware.
- **Advanced channel model** – LOS links use Rician fading, NLOS links fall back to Rayleigh/Nakagami with optional log-normal shadowing.
- **Queue-based latency** – each UE maintains a backlog (in Mbits); latency is derived from queue drain time vs. latency budget instead of a heuristic ratio.
- **Schedulers** – proportional-fair scheduling is enabled by default (round-robin is available via `scheduler_mode`), and base-station load is derived from requested traffic vs. capacity.

## RL API (Phase 2)

`environment.py` now doubles as an RL-ready simulator:

- `reset_for_rl()` → resets mobility/load history and returns the stacked observation matrix plus each UE's candidate BS map.
- `build_state_matrix()` → can be called anytime to get the current observations without stepping the sim.
- `step_with_actions(actions)` → moves UEs, applies a discrete BS-selection action per UE, and returns `(next_states, rewards, info)` where `info["metrics"]` stores per-UE stats and `info["candidate_map"]` feeds the next action pass. Rewards balance throughput gains against overload, latency-budget violations, and handover costs (weights configurable via `DEFAULT_RL_CONFIG`).

**State layout:** `[demand_norm, backlog_norm, latency_budget_norm, latency_slack_norm, last_handover_flag]` followed by, for each of the top-`k` candidate base stations (default `k=2`), `[normalized signal strength, normalized previous load, normalized capacity]`. All values are clipped to `[0, 1]` (or `[-1, 1]` for latency slack) so a neural agent can ingest them directly.

**Action space:** one discrete choice per UE (index into its candidate list). Invalid actions are clipped to the closest candidate. Set `num_candidates` in `DEFAULT_RL_CONFIG` to control action dimensionality.

**Reward:** `w_t * throughput_ratio - w_o * overload - w_l * latency_violation - w_q * queue_norm - w_s * sinr_deficit - w_h * handover_flag`, with tunable weights under `reward_weights`. Additional realism knobs live in `DEFAULT_RL_CONFIG` too (`timestep_s`, `scheduler_mode` for PF vs. round-robin resource allocation, `sinr_target_db`).

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
- Tweak exploration or reward weights by editing `TrainConfig` / `DEFAULT_RL_CONFIG` (e.g., handover and SINR penalties). When the state layout changes, make sure to retrain before evaluating old checkpoints.

### Evaluating a trained DQN

```bash
python main.py --dqn-weights artifacts/dqn_weights.pt --steps 60
```

This runs the baseline, load-aware heuristic, and the loaded DQN policy side by side, prints the KPI summary for each, and plots all histories (use `--no-plots` if running remotely).

## Custom topologies

`topology.py` now accepts JSON specifications so you can change cell layouts or UE placements without editing code.

### Spec format (see `topologies/hotspot_dense.json`)

```json
{
  "area_size": 120,
  "num_users": 60,
  "base_stations": [
    {"bs_id": 0, "x": 10, "y": 20, "tier": "macro"},
    {"bs_id": 1, "x": 90, "y": 25, "tier": "macro"},
    {"bs_id": 2, "x": 60, "y": 100, "tier": "macro"},
    {"bs_id": 3, "x": 35, "y": 50, "tier": "micro"}
  ],
  "user_positions": [
    {"x": 30, "y": 45, "traffic_profile": "urllc"},
    {"x": 70, "y": 65, "traffic_profile": "embb"}
  ]
}
```

- `base_stations`: required; each entry can optionally override `capacity_mbps`, `tx_power_dbm`, `bandwidth_mhz`, or `resource_blocks`.
- `num_users` + `area_size`: used for deterministic UE seeding when `user_positions` is omitted.
- `user_positions`: optional; when present, the UE count equals the length of this list (each entry may set a `traffic_profile`).

### Using a custom spec

- **CLI sim:** `python main.py --topology ../topologies/hotspot_dense.json`
- **DQN eval:** `python main.py --topology ../topologies/hotspot_dense.json --dqn-weights artifacts/dqn_weights.pt`
- **FastAPI service:** set `TOPOLOGY_FILE` before launching, e.g. `TOPOLOGY_FILE=../topologies/hotspot_dense.json python ric_api.py`
- **Docker Compose:** mount the spec into the API container and set `environment: TOPOLOGY_FILE=/app/hotspot.json`.

All downstream components (xApp agent, dashboard) automatically consume whatever topology the API exposes.

## Latency-Aware xApp Mini Stack

To mirror a near-RT RIC + xApp control loop, the repo now ships with:

- `ric_api.py` – FastAPI service that keeps the simulator running, serves `/metrics` snapshots (cell loads, UE latency/throughput/SINR, normalized RL states, candidate BS IDs) and accepts `/actions` posts with handover directives.
- `xapp_agent.py` – CLI loop that polls the API and dispatches actions using either the built-in latency-aware heuristic or a trained DQN checkpoint.

Architecture:

```
[NetworkEnvironment + topology] -> (FastAPI /metrics) -> [xApp agent logic] -> (POST /actions) -> [Simulator control loop]
```

### Running the stack

1. Start the metrics service (from `ai_5g_load_balancing/`):
   ```bash
   python ric_api.py  # or uvicorn ric_api:app --host 0.0.0.0 --reload
   ```
   - `GET /metrics?advance=true` steps the sim via the heuristic controller and returns the latest snapshot (UE metrics, per-cell load, RL states, candidate map as BS IDs).
   - `POST /actions` expects `{"actions": [{"ue_id": 3, "target_bs": 1}, ...]}`. Unspecified UEs fall back to the heuristic, so partial updates are fine.
   - `GET /history?tail=30` surfaces recent KPI traces for plotting dashboards.

2. Launch the xApp client (heuristic mode by default):
   ```bash
   python xapp_agent.py --api-base http://127.0.0.1:8000 --steps 50 --interval 1.0
   ```
   - Sends up to `--max-actions` latency-motivated handovers per step, skipping cells whose load exceeds `--load-threshold`.
   - Logs KPI deltas each iteration so you can quickly compare before/after latency and throughput.

3. (Optional) Run the xApp in DQN mode with a trained checkpoint:
   ```bash
   python xapp_agent.py --mode dqn --dqn-weights artifacts/dqn_weights.pt --steps 60
   ```
   - Reuses the normalized observation matrix served by `/metrics` and maps Q-network argmax decisions back to BS IDs via the candidate map before posting them.

This “mini xApp” workflow gives you a concrete artifact to talk about: a controller microservice that ingests live RAN metrics and applies policy-driven load-balancing actions, ready to be containerized for Kubernetes demos later on.

### Logging

- The FastAPI service emits step summaries and action events to `logs/ric_api.log` (rotating 1 MB × 5 files). Tail this file while running locally to observe controller decisions; the `logs/` folder is already gitignored and excluded from Docker builds.

### Containerized demo

- `Dockerfile.ric_api`, `Dockerfile.xapp`, and `Dockerfile.dashboard` build slim Python 3.12 images for the FastAPI service, xApp loop, and Streamlit dashboard respectively. A shared `.dockerignore` keeps virtualenvs and artifacts out of the build context.
- `docker-compose.yml` wires everything together on a private network, publishes the API on `localhost:8000`, the dashboard on `localhost:8501`, mounts `./ai_5g_load_balancing/artifacts` into every container so DQN weights can be shared, **and** maps `./topologies/hotspot_dense.json` into the API container while setting `TOPOLOGY_FILE=/app/topology.json`. Swap this bind mount + env var to test other specs.

Bring the whole stack up with:

```bash
docker compose up --build
```

The xApp service defaults to the heuristic policy (override `command` in `docker-compose.yml` for DQN mode), and the dashboard automatically points at `http://ric-api:8000`.

### Streamlit dashboard

`dashboard.py` renders `/metrics` summaries, `/history` trendlines (latency, throughput, cell load, queues), and a live topology scatter plot overlaying base stations/UEs plus SINR-colored UE→BS link lines so you can literally see the “flow” of signals. Run it locally with:

```bash
streamlit run dashboard.py --server.port 8501
```

Use the sidebar to change the API base URL (defaults to `http://127.0.0.1:8000` or `API_BASE` env var), adjust the history window, and trigger manual refreshes. The “Top latency-challenged UEs” table highlights candidates for handover, while the charts show how the xApp policies are affecting KPIs over time.
