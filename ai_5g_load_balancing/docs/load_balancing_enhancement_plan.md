# Load-Balancing Logic Enhancement Plan

This plan focuses on evolving the association logic under `controller.py` and the RL-facing pathways in `environment.py` / `rl_models.py`. The goal is to produce measurable throughput and latency improvements while containing handovers and congestion.

## 1. Profile Existing Heuristics and RL Policies

**Objectives**
- Quantify how `strongest_signal_policy`, `load_aware_policy`, and `predictive_mobility_policy` perform across representative traffic mixes and mobility patterns.
- Capture fairness (per-user throughput), latency conformance, and cell overloading under stress (burst demand, hotspots, velocity spikes).

**Actions**
1. **Scenario matrix**: use `topologies/hotspot_dense.json` plus the default topology and vary user counts (25, 50, 75) and demand-shaper seeds (set via `np.random.seed` before `env.run`).
2. **Profiling harness**: add `scripts/profile_policies.py` that:
   - Builds the network via `topology.build_reference_network` or JSON spec.
   - Runs each policy for N=200 steps via `NetworkEnvironment.run`.
   - Persists `env.history` plus derived metrics (Jain fairness index, 95th percentile latency, overload violations) to `reports/load_balancing_profile_<timestamp>.json`.
3. **Telemetry hooks**: extend `NetworkEnvironment.history` with:
   - `fairness_index`, calculated per step.
   - `latency_violation_rate` (share of UEs exceeding `latency_budget_ms`).
   - `cell_overload_flags` (count of BS with `load > 1.2`).
4. **Visualization**: optionally reuse `main.py` plotting helpers; add CLI flag `--profile-report path.json` to dump aggregated stats without plots.

## 2. Define Performance Targets

Use profiling data to lock targets that the adaptive policy must meet:

| Metric | Baseline Observation (estimate) | Target |
| --- | --- | --- |
| Avg throughput (Mbps) | 8–10 (strongest signal), 11–12 (load-aware) | ≥ 13 while keeping fairness ≥ 0.9 |
| Jain fairness index | 0.78 (strongest signal), 0.86 (load-aware) | ≥ 0.9 |
| 95p latency (ms) | 120–150 | ≤ 100 |
| Cell overload ratio (>1.0 load) | 28% of steps | ≤ 15% |
| Handovers / step | 2.8 (predictive mobility) | ≤ 3 with no >20% spikes during demand surges |

Document these targets inside the repo (e.g., `docs/perf_targets.json`) so automated tests can assert against them.

## 3. Prototype Adaptive Association Policy

**Concept**
- Use a two-stage decision: (1) candidate ranking via predictive features, (2) adaptive controller that blends heuristic and RL outputs.
- Features already available in `NetworkEnvironment._build_state_vector` (demand/backlog/latency/speed, per-BS signal/load/capacity) cover the needed context.

**Implementation Steps**
1. **Policy façade**: create `controller.AdaptivePolicyManager` that exposes:
   - `update_context(env_metrics)` every step to ingest load/latency history.
   - `select_bs(ue, base_stations)` returning BS id.
2. **Multi-armed scoring**:
   - Compute heuristic scores (signal, load-aware, predictive mobility).
   - Retrieve RL-proposed actions when `rl_models` are available (optional weights path).
   - Blend via adaptive weights `w = softmax(theta · context)` learned online (e.g., contextual bandit) or configured heuristically.
3. **Short-horizon prediction**:
   - Estimate UE dwell time and bs backlog slope (already partially in `predictive_mobility_policy`); expose helper functions so both heuristics and adaptive policy reuse them.
4. **Learning loop (optional)**:
   - Use lightweight contextual bandit: maintain per-policy reward estimates keyed by (traffic_profile, mobility_profile, demand regime bucket). Update after `_finalize_step` using throughput - penalty formula similar to RL reward weights.
5. **Configuration**:
   - Introduce `config/policy.yaml` to toggle adaptive behavior, RL weights path, and fallback policy.

Deliverables for this phase: new policy class, updated `main.py` switch (`--policy adaptive`), config plumbing, and docstrings.

## 4. Simulation Benchmarks

1. **Batch runner**: add `scripts/run_benchmarks.py --policy <name> --topology <path> --steps 300 --seeds 5` that aggregates metrics (mean ± std) and emits Markdown/JSON tables.
2. **KPIs to log**: throughput mean/95p, latency mean/95p, fairness, overload rate, handover totals, SINR mean, queue backlog.
3. **Regression gate**: keep last-known-good benchmark snapshot in `reports/baseline_metrics.json`. CI job compares new runs; fails when targets regress beyond tolerance (e.g., throughput drop >5%).

## 5. Tests & Automation

- **Unit tests**:
  - Validate `AdaptivePolicyManager` weight updates and tie-breaking using deterministic mock UEs/BS.
  - Test telemetry helpers (fairness, overload flags) with synthetic data.
- **Integration tests**:
  - Short deterministic simulation (seeded) verifying the adaptive policy meets mini targets (e.g., fairness >= load-aware baseline, overload rate <= baseline).
  - CLI smoke test: `python -m ai_5g_load_balancing.main --policy adaptive --steps 5 --no-plots` runs without GPU / weights.
- **CI wiring**: add GitHub workflow job that runs unit tests + quick benchmark (steps=30) on every PR touching `controller.py`, `environment.py`, or `rl_models.py`.

## 6. Timeline & Dependencies

1. **Day 1–2**: Instrumentation + profiling harness, capture baseline reports.
2. **Day 3**: Finalize performance targets from reports, author config files.
3. **Day 4–6**: Implement adaptive policy manager + configs + CLI hooks.
4. **Day 7**: Build benchmark runner and regression snapshots.
5. **Day 8**: Add tests/CI, refine documentation (README + dashboards).

Dependencies: `numpy`, `torch` (if RL blending enabled), plotting stack for offline analysis. Ensure `.venv` activated and add any new packages to `requirements.txt`.
