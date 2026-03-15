import numpy as np

CARRIER_FREQ_GHZ = 3.5
BANDWIDTH_MHZ = 20
NOISE_FLOOR_DBM = -104  # thermal noise for ~20 MHz channel
PATH_LOSS_EXP = 3.5

# Fading configuration defaults
FADING_CONFIG = {
    "fast_model": "rayleigh",  # rayleigh, rician, nakagami
    "rician_k": 4.0,
    "nakagami_m": 1.5,
    "shadowing_sigma_db": 6.0,
    "enable_shadowing": True,
}


def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def path_loss_db(distance_m, path_loss_exp=PATH_LOSS_EXP):
    distance_m = max(distance_m, 1.0)
    reference_distance_m = 1.0
    loss = 10 * path_loss_exp * np.log10(distance_m / reference_distance_m)
    carrier_component = 20 * np.log10(CARRIER_FREQ_GHZ)
    free_space_term = 32.4  # Friis constant in dB
    return free_space_term + carrier_component + loss


def received_power_dbm(tx_power_dbm, distance_m):
    return tx_power_dbm - path_loss_db(distance_m)


def signal_strength(ue, bs):
    """Return received power in dBm (used by association policies)."""
    d = distance(ue.x, ue.y, bs.x, bs.y)
    return received_power_dbm(bs.tx_power_dbm, d)


def dbm_to_mw(dbm):
    return 10 ** (dbm / 10)


def sample_rayleigh_power():
    sigma = 1 / np.sqrt(2)  # ensures unit-mean power
    amplitude = np.random.rayleigh(scale=sigma)
    return amplitude**2


def sample_rician_power(k_factor):
    k_factor = max(k_factor, 0.0)
    sigma = np.sqrt(1 / (2 * (k_factor + 1)))
    los = np.sqrt(k_factor / (k_factor + 1))
    x = np.random.normal(los, sigma)
    y = np.random.normal(0.0, sigma)
    amplitude = np.sqrt(x**2 + y**2)
    return amplitude**2


def sample_nakagami_power(m, omega=1.0):
    m = max(m, 0.1)
    return np.random.gamma(shape=m, scale=omega / m)


def sample_shadowing_gain(sigma_db):
    sigma_db = max(sigma_db, 0.0)
    offset_db = np.random.normal(0.0, sigma_db)
    return 10 ** (offset_db / 10)


def sample_fading_gain(config=None):
    cfg = config if config else FADING_CONFIG
    model = cfg.get("fast_model", "rayleigh").lower()
    if model == "rician":
        fast = sample_rician_power(cfg.get("rician_k", 4.0))
    elif model == "nakagami":
        fast = sample_nakagami_power(cfg.get("nakagami_m", 1.5))
    else:
        fast = sample_rayleigh_power()

    if cfg.get("enable_shadowing", True):
        fast *= sample_shadowing_gain(cfg.get("shadowing_sigma_db", 6.0))
    return fast


def sinr_linear(ue, serving_bs, base_stations, fading_config=None):
    signal_dbm = received_power_dbm(
        serving_bs.tx_power_dbm,
        distance(ue.x, ue.y, serving_bs.x, serving_bs.y),
    )
    signal_mw = dbm_to_mw(signal_dbm) * sample_fading_gain(fading_config)

    interference_mw = 0.0
    for other in base_stations:
        if other.bs_id == serving_bs.bs_id:
            continue
        power_dbm = received_power_dbm(
            other.tx_power_dbm,
            distance(ue.x, ue.y, other.x, other.y),
        )
        interference_mw += dbm_to_mw(power_dbm) * sample_fading_gain(
            fading_config
        )

    noise_mw = dbm_to_mw(NOISE_FLOOR_DBM)
    denominator = noise_mw + interference_mw
    if denominator <= 0:
        denominator = noise_mw
    return signal_mw / denominator


def shannon_throughput_mbps(sinr_value, bandwidth_mhz=BANDWIDTH_MHZ):
    spectral_eff = np.log2(1 + sinr_value)
    return (bandwidth_mhz * 1e6 * spectral_eff) / 1e6  # convert back to Mbps


def user_throughput(ue, bs, sinr_value):
    radio_rate = shannon_throughput_mbps(sinr_value)
    offered = min(ue.demand, radio_rate)
    if bs.load <= 1.0:
        return offered
    return offered / bs.load


def estimate_latency_ms(ue, throughput_mbps):
    if throughput_mbps <= 1e-6:
        return ue.latency_budget_ms * 5
    utilization = min(ue.demand / throughput_mbps, 5.0)
    queue_penalty = max(utilization, 1.0)
    return ue.latency_budget_ms * queue_penalty
