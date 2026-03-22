import numpy as np

NOISE_FLOOR_DBM = -104  # thermal noise reference

# Fading configuration defaults
FADING_CONFIG = {
    "los_model": "rician",
    "nlos_model": "rayleigh",
    "rician_k": 6.0,
    "nakagami_m": 1.5,
    "shadowing_sigma_db": 6.0,
    "enable_shadowing": True,
}


def link_distance(ue, bs):
    bs_height = getattr(bs, "height_m", 0.0)
    ue_z = getattr(ue, "z", 0.0)
    ue_height = getattr(ue, "height_m", 1.5) if hasattr(ue, "height_m") else 1.5
    x_term = (ue.x - bs.x) ** 2
    y_term = (ue.y - bs.y) ** 2
    z_term = (ue_z + ue_height - (bs.z + bs_height)) ** 2
    return np.sqrt(x_term + y_term + z_term)


def los_probability(distance_m, tier, environment=None):
    distance_m = max(distance_m, 1.0)
    if tier == "micro":
        base = 0.9 if distance_m <= 18 else max(0.1, np.exp(-(distance_m - 18) / 36))
    else:
        base = 0.95 if distance_m <= 30 else max(
            0.05, np.exp(-(distance_m - 30) / 80)
        )
    if environment == "indoor":
        base *= 0.6
    return float(np.clip(base, 0.01, 0.99))


def _legacy_path_loss(distance_m, params, is_los):
    exp = params["los_exp"] if is_los else params["nlos_exp"]
    intercept = params["los_intercept"] if is_los else params["nlos_intercept"]
    distance_m = max(distance_m, 1.0)
    return intercept + 10 * exp * np.log10(distance_m)


def _free_space_loss(distance_m, frequency_ghz, **_):
    distance_km = max(distance_m / 1000.0, 1e-3)
    freq_mhz = max(frequency_ghz * 1000.0, 1.0)
    return 32.45 + 20.0 * np.log10(distance_km) + 20.0 * np.log10(freq_mhz)


def _cost231_loss(distance_m, frequency_ghz, bs, ue_height=1.5, **kwargs):
    params = kwargs.get("params", {})
    distance_km = max(distance_m / 1000.0, 1e-3)
    freq_mhz = max(frequency_ghz * 1000.0, 150.0)
    hb = max(getattr(bs, "height_m", 25.0), 5.0)
    hm = max(ue_height, 1.0)
    c_env = params.get("environment_correction_db", 3.0)
    a_hm = (1.1 * np.log10(freq_mhz) - 0.7) * hm - (1.56 * np.log10(freq_mhz) - 0.8)
    return (
        46.3
        + 33.9 * np.log10(freq_mhz)
        - 13.82 * np.log10(hb)
        - a_hm
        + (44.9 - 6.55 * np.log10(hb)) * np.log10(distance_km)
        + c_env
    )


def _nyu_mmwave_loss(distance_m, frequency_ghz, is_los=True, **kwargs):
    params = kwargs.get("params", {})
    n_los = params.get("los_exp", 2.0)
    n_nlos = params.get("nlos_exp", 3.2)
    sigma_los = params.get("los_sigma_db", 4.0)
    sigma_nlos = params.get("nlos_sigma_db", 7.0)
    intercept = 32.4
    path_exp = n_los if is_los else n_nlos
    sigma = sigma_los if is_los else sigma_nlos
    distance_m = max(distance_m, 1.0)
    return (
        intercept
        + 10 * path_exp * np.log10(distance_m)
        + 20 * np.log10(max(frequency_ghz, 0.1))
        + np.random.normal(0.0, sigma)
    )


PATH_LOSS_MODELS = {
    "free_space": _free_space_loss,
    "cost231": _cost231_loss,
    "nyu_mmwave": _nyu_mmwave_loss,
}


def path_loss_db(distance_m, bs, carrier, ue_height=1.5, is_los=True):
    params = carrier.path_loss_params or bs.path_loss_params or {}
    model_name = (carrier.path_loss_model or bs.path_loss_model or "free_space").lower()
    if {"los_exp", "nlos_exp", "los_intercept", "nlos_intercept"} <= params.keys():
        return _legacy_path_loss(distance_m, params, is_los=is_los)
    model_fn = PATH_LOSS_MODELS.get(model_name, _free_space_loss)
    return model_fn(
        distance_m=distance_m,
        frequency_ghz=getattr(carrier, "frequency_ghz", 3.5),
        bs=bs,
        ue_height=ue_height,
        params=params,
        is_los=is_los,
    )


def received_power_dbm(ue, bs, carrier, is_los=True):
    dist = link_distance(ue, bs)
    ue_height = getattr(ue, "height_m", 1.5)
    loss = path_loss_db(dist, bs, carrier, ue_height=ue_height, is_los=is_los)
    return carrier.tx_power_dbm - loss


def signal_strength(ue, bs):
    """Return expected LOS received power in dBm for the BS primary carrier."""
    carrier = bs.primary_carrier
    return received_power_dbm(ue, bs, carrier, is_los=True)


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


def _match_carrier(other_bs, carrier):
    match = other_bs.get_carrier(carrier.name)
    if match:
        return match
    for cand in other_bs.iter_carriers():
        if abs(cand.frequency_ghz - carrier.frequency_ghz) <= 0.25:
            return cand
    return None


def sinr_linear(ue, serving_bs, base_stations, carrier, fading_config=None):
    d_serv = link_distance(ue, serving_bs)
    los_serv = np.random.rand() < los_probability(
        d_serv, serving_bs.tier, environment=getattr(ue, "environment", None)
    )
    signal_dbm = received_power_dbm(ue, serving_bs, carrier, is_los=los_serv)
    signal_mw = dbm_to_mw(signal_dbm) * sample_fading_gain_link(
        los_serv, fading_config
    )

    interference_mw = 0.0
    for other in base_stations:
        if other.bs_id == serving_bs.bs_id:
            continue
        other_carrier = _match_carrier(other, carrier)
        if other_carrier is None:
            continue
        d_int = link_distance(ue, other)
        los_int = np.random.rand() < los_probability(
            d_int, other.tier, environment=getattr(ue, "environment", None)
        )
        power_dbm = received_power_dbm(ue, other, other_carrier, is_los=los_int)
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


def phy_rate_from_rbs(bandwidth_mhz, resource_blocks, cqi_efficiency, allocated_rbs):
    if allocated_rbs <= 0:
        return 0.0
    total_rbs = max(resource_blocks, 1)
    rb_bw_hz = (bandwidth_mhz * 1e6) / total_rbs
    rb_rate_mbps = cqi_efficiency * rb_bw_hz / 1e6
    return rb_rate_mbps * allocated_rbs


def user_throughput_from_phy(
    ue,
    bandwidth_mhz,
    resource_blocks,
    sinr_val,
    allocated_rbs,
    timestep_s=1.0,
):
    sinr_db = 10 * np.log10(sinr_val + 1e-9)
    cqi = sinr_to_cqi(sinr_db)
    eff = cqi_to_efficiency(cqi)
    phy_rate = phy_rate_from_rbs(bandwidth_mhz, resource_blocks, eff, allocated_rbs)
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
