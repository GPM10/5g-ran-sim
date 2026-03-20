import numpy as np

NOISE_FLOOR_DBM = -104  # thermal noise for ~20 MHz channel

# Fading configuration defaults
FADING_CONFIG = {
    "los_model": "rician",
    "nlos_model": "rayleigh",
    "rician_k": 6.0,
    "nakagami_m": 1.5,
    "shadowing_sigma_db": 6.0,
    "enable_shadowing": True,
}


def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def los_probability(distance_m, tier):
    distance_m = max(distance_m, 1.0)
    if tier == "micro":
        if distance_m <= 18:
            return 0.9
        return max(0.1, np.exp(-(distance_m - 18) / 36))
    # macro default
    if distance_m <= 30:
        return 0.95
    return max(0.05, np.exp(-(distance_m - 30) / 80))


def path_loss_db(distance_m, bs, is_los=True):
    params = bs.path_loss_params
    exp = params["los_exp"] if is_los else params["nlos_exp"]
    intercept = params["los_intercept"] if is_los else params["nlos_intercept"]
    distance_m = max(distance_m, 1.0)
    return intercept + 10 * exp * np.log10(distance_m)


def received_power_dbm(bs, distance_m, is_los=True):
    return bs.tx_power_dbm - path_loss_db(distance_m, bs, is_los=is_los)


def signal_strength(ue, bs):
    """Return expected LOS received power in dBm (used by association policies)."""
    d = distance(ue.x, ue.y, bs.x, bs.y)
    return received_power_dbm(bs, d, is_los=True)


def dbm_to_mw(dbm):
    return 10 ** (dbm / 10)


def sample_rayleigh_power():
    sigma = 1 / np.sqrt(2)
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


def sample_fading_gain_link(is_los, config=None):
    cfg = config if config else FADING_CONFIG
    model = cfg.get("los_model" if is_los else "nlos_model", "rayleigh").lower()
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
    d_serv = distance(ue.x, ue.y, serving_bs.x, serving_bs.y)
    los_serv = np.random.rand() < los_probability(d_serv, serving_bs.tier)
    signal_dbm = received_power_dbm(serving_bs, d_serv, is_los=los_serv)
    signal_mw = dbm_to_mw(signal_dbm) * sample_fading_gain_link(
        los_serv, fading_config
    )

    interference_mw = 0.0
    for other in base_stations:
        if other.bs_id == serving_bs.bs_id:
            continue
        d_int = distance(ue.x, ue.y, other.x, other.y)
        los_int = np.random.rand() < los_probability(d_int, other.tier)
        power_dbm = received_power_dbm(other, d_int, is_los=los_int)
        interference_mw += dbm_to_mw(power_dbm) * sample_fading_gain_link(
            los_int, fading_config
        )

    noise_mw = dbm_to_mw(NOISE_FLOOR_DBM)
    denominator = noise_mw + interference_mw
    if denominator <= 0:
        denominator = noise_mw
    return signal_mw / denominator


def shannon_throughput_mbps(sinr_value, bandwidth_mhz):
    spectral_eff = np.log2(1 + sinr_value)
    return (bandwidth_mhz * 1e6 * spectral_eff) / 1e6
CQI_THRESHOLDS_DB = [
    -6.7,
    -4.7,
    -2.3,
    0.0,
    1.0,
    3.0,
    5.0,
    7.0,
    9.0,
    11.0,
    12.7,
    14.7,
    16.7,
    18.7,
    21.0,
]

MCS_EFFICIENCY = [
    0.1523,
    0.2344,
    0.3770,
    0.6016,
    0.8770,
    1.1758,
    1.4766,
    1.9141,
    2.4063,
    2.7305,
    3.3223,
    3.9023,
    4.5234,
    5.1152,
    5.5547,
    5.8984,
]


def sinr_to_cqi(sinr_db):
    for idx, threshold in enumerate(CQI_THRESHOLDS_DB):
        if sinr_db < threshold:
            return max(idx, 0)
    return len(CQI_THRESHOLDS_DB)


def cqi_to_efficiency(cqi_idx):
    cqi_idx = int(np.clip(cqi_idx, 0, len(MCS_EFFICIENCY) - 1))
    return MCS_EFFICIENCY[cqi_idx]


def phy_rate_from_rbs(bs, cqi_efficiency, allocated_rbs):
    if allocated_rbs <= 0:
        return 0.0
    rb_bw_hz = (bs.bandwidth_mhz * 1e6) / max(bs.resource_blocks, 1)
    rb_rate_mbps = cqi_efficiency * rb_bw_hz / 1e6
    return rb_rate_mbps * allocated_rbs


def user_throughput_from_phy(ue, bs, sinr_val, allocated_rbs, timestep_s=1.0):
    sinr_db = 10 * np.log10(sinr_val + 1e-9)
    cqi = sinr_to_cqi(sinr_db)
    eff = cqi_to_efficiency(cqi)
    phy_rate = phy_rate_from_rbs(bs, eff, allocated_rbs)
    backlog_rate = ue.backlog_mbits / max(timestep_s, 1e-6)
    desired_rate = max(backlog_rate, ue.demand)
    served_rate = min(phy_rate, desired_rate)
    served_data = served_rate * timestep_s
    ue.backlog_mbits = max(ue.backlog_mbits - served_data, 0.0)
    return served_rate, phy_rate, cqi


def estimate_latency_ms(ue, throughput_mbps, backlog_before_mbits, timestep_s=1.0):
    if throughput_mbps <= 1e-6:
        return ue.latency_budget_ms * 5
    wait_seconds = backlog_before_mbits / max(throughput_mbps, 1e-3)
    latency = min(wait_seconds * 1000.0, ue.latency_budget_ms * 5)
    return latency
