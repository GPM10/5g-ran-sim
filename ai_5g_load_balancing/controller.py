from utils import signal_strength


def strongest_signal_policy(ue, base_stations):
    return max(base_stations, key=lambda bs: signal_strength(ue, bs))


def load_aware_policy(ue, base_stations, alpha=0.7, beta=0.3):
    """
    Simple AI-like heuristic:
    score = alpha * normalized signal - beta * current load
    """
    signals = [signal_strength(ue, bs) for bs in base_stations]
    max_sig = max(signals) if max(signals) > 0 else 1.0

    best_bs = None
    best_score = -1e9

    for bs, sig in zip(base_stations, signals):
        score = alpha * (sig / max_sig) - beta * bs.load
        if score > best_score:
            best_score = score
            best_bs = bs

    return best_bs
