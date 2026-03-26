import math

from .utils import link_distance, signal_strength


def _transport_penalty(bs, context):
    if not context:
        return 0.0
    penalty = 0.0
    bs_to_du = context.get("bs_to_du", {})
    du_id = bs_to_du.get(bs.bs_id)
    if du_id:
        stats = context.get("f1", {}).get(du_id, {})
        user_queue = stats.get("user_queue", 0.0)
        control_queue = stats.get("control_queue", 0.0)
        penalty += min(user_queue / 20.0, 1.5)
        penalty += 0.2 * min(control_queue / 10.0, 1.0)
    ru_stats = context.get("ru", {}).get(bs.bs_id, {})
    temp_pen = max(ru_stats.get("temperature_c", 40.0) - 70.0, 0.0) / 40.0
    penalty += temp_pen
    core = context.get("core", {})
    upf_pen = max(core.get("upf_load", 0.0) - 0.85, 0.0)
    penalty += upf_pen
    return penalty


def _intent_multiplier(ue, context):
    if not context:
        return 1.0
    intents = context.get("intents", {})
    intent = intents.get(getattr(ue, "traffic_profile", ""), None)
    if not intent:
        return 1.0
    priority = max(intent.get("priority", 1), 1)
    return 1.0 + 0.15 * (priority - 1)


def strongest_signal_policy(ue, base_stations, context=None):
    return max(base_stations, key=lambda bs: signal_strength(ue, bs))


def load_aware_policy(ue, base_stations, context=None, alpha=0.7, beta=0.3):
    """
    Simple AI-like heuristic:
    score = alpha * normalized signal - beta * current load
    """
    signals = [signal_strength(ue, bs) for bs in base_stations]
    max_sig = max(signals) if max(signals) > 0 else 1.0

    best_bs = None
    best_score = -1e9

    intent_boost = _intent_multiplier(ue, context)

    for bs, sig in zip(base_stations, signals):
        transport_pen = _transport_penalty(bs, context)
        score = intent_boost * alpha * (sig / max_sig) - beta * bs.load - 0.3 * transport_pen
        if score > best_score:
            best_score = score
            best_bs = bs

    return best_bs


ADVANCED_WEIGHTS = {
    "w_signal": 0.35,
    "w_load": 0.25,
    "w_capacity": 0.15,
    "w_latency": 0.15,
    "w_mobility": 0.07,
    "w_stability": 0.03,
    "w_velocity": 0.05,
    "load_cap": 2.5,
    "capacity_cap": 200.0,
    "velocity_cap": 15.0,
    "dwell_horizon_s": 4.0,
}


def predictive_mobility_policy(ue, base_stations, context=None, weights=None):
    """
    Advanced heuristic that mixes RF quality, cell load, UE latency pressure,
    and short-horizon mobility. It prefers cells the UE is moving toward,
    with spare capacity to drain the current backlog within the SLA.
    """
    cfg = {**ADVANCED_WEIGHTS, **(weights or {})}
    heading_rad = math.radians(getattr(ue, "heading_deg", 0.0))
    heading_vec = (math.cos(heading_rad), math.sin(heading_rad))
    velocity = max(getattr(ue, "velocity_m_s", 0.0), 0.0)
    max_signal = max(signal_strength(ue, bs) for bs in base_stations) or 1.0
    latency_budget_s = max(getattr(ue, "latency_budget_ms", 50.0) / 1000.0, 1e-3)
    demand_rate = max(getattr(ue, "demand", 0.1), 0.1)
    backlog = max(getattr(ue, "backlog_mbits", 0.0), 0.0)
    latency_pressure = min(backlog / max(latency_budget_s * demand_rate, 1e-3), 4.0)

    best_bs = None
    best_score = float("-inf")

    intent_boost = _intent_multiplier(ue, context)

    for bs in base_stations:
        sig_norm = signal_strength(ue, bs) / max_signal
        load_term = 1.0 - min(bs.load, cfg["load_cap"]) / cfg["load_cap"]
        capacity_term = min(bs.capacity_mbps, cfg["capacity_cap"]) / cfg["capacity_cap"]
        latency_term = latency_pressure * capacity_term

        distance = max(link_distance(ue, bs), 1.0)
        dir_vec = ((bs.x - ue.x) / distance, (bs.y - ue.y) / distance)
        alignment = (heading_vec[0] * dir_vec[0] + heading_vec[1] * dir_vec[1] + 1.0) / 2.0
        dwell = math.exp(
            -distance / max(cfg["dwell_horizon_s"] * max(velocity, 0.5), 1.0)
        )
        mobility_term = alignment * dwell

        velocity_term = min(velocity / cfg["velocity_cap"], 1.0) * mobility_term
        stability_term = 1.0 if getattr(ue, "serving_bs", None) == bs.bs_id else 0.0

        penalty = _transport_penalty(bs, context)

        score = intent_boost * (
            cfg["w_signal"] * sig_norm
            + cfg["w_load"] * load_term
            + cfg["w_capacity"] * capacity_term
            + cfg["w_latency"] * latency_term
            + cfg["w_mobility"] * mobility_term
            + cfg["w_velocity"] * velocity_term
            + cfg["w_stability"] * stability_term
        ) - 0.5 * penalty
        if score > best_score:
            best_score = score
            best_bs = bs

    return best_bs or strongest_signal_policy(ue, base_stations)
