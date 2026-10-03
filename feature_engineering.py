"""
feature_engineering.py
======================
Exact reproduction of the feature engineering from Moustafa's own training
script (unified multi-fault benchmark, CF/EO/RL/FWE/FWC + OC/NC). Replaces
the earlier reconstructed/unverified version now that the real source was
provided.

Key facts taken directly from the source script (not guessed):

  - All temperatures are converted to KELVIN before any formula is applied
    (RP-1043: degF -> K; EnergyPlus/SQL: degC -> K). This matters: Kelvin
    vs Celsius differences are identical (same magnitude), but the RAW
    temperature values used in ratios like x1 = T_ci_K / Q_evap_kW are NOT
    the same number in Kelvin vs Celsius, and the trained models expect Kelvin.

  - T_ci ("condenser inlet proxy") is dataset-dependent:
      * RP-1043: real condenser water inlet temperature (TWCI column).
      * EnergyPlus/SQL: NO condenser-side variable was pulled from the SQL
        file at all (see get_var_ids - only evaporator inlet/outlet, power,
        Qevap, COP). The script sets T_ci_K = T_ei_K (evaporator inlet) as
        a stand-in. This looks unusual but is exactly what the uploaded
        svm_*_sql.pkl / physics_*_sql.pkl models were trained on, so it is
        reproduced as-is here for the "sql" dataset.

  - The physics (Gordon-Ng) model predicts y = Power_kW / Q_evap_kW
    (i.e. inverse COP), not Power directly - predicted power is recovered
    as Q_evap_kW * model.predict(x1, x2).

  - Power_Residual_norm is normalized by Q_evap_kW, NOT by predicted power.

  - Heat_rejection_ratio and Cond_dT are literally the same formula
    (Q_cond / Q_evap) - confirmed duplicate, kept as-is.

  - Evap_approach falls back to the same formula as Evap_effectiveness
    when no evaporator-inlet temperature is available; for EnergyPlus/SQL
    data this fallback always applies (T_ei_K is the only inlet variable
    EnergyPlus exposes, and it's already used as the T_ci proxy upstream
    of this calculation in the original script's per-row dataframe, but
    Evap_approach's own check looks at the same T_ei_K column, so it takes
    the real branch: Evap_approach = T_ei_K - T_eo_K for both datasets as
    long as T_ei_K is populated).
"""

from dataclasses import dataclass

EPS = 1e-3

FEATURE_ORDER = [
    "Power_Residual",
    "Power_Residual_norm",
    "COP",
    "Power_per_load",
    "delta_T_lift",
    "delta_T_norm",
    "Evap_effectiveness",
    "COP_ratio",
    "Heat_rejection_ratio",
    "Comp_work_ratio",
    "Cond_dT",
    "Evap_approach",
]


@dataclass
class RawReading:
    """Raw sensor snapshot, ALREADY IN THE SAME UNITS THE ORIGINAL SCRIPT
    USED: temperatures in degrees Celsius (converted to Kelvin internally,
    matching both the RP-1043 degF->K and the SQL degC->K paths), power
    and cooling rate in kW.

    dataset: "sql" (EnergyPlus) or "rp1043" (ASHRAE RP-1043) - changes how
    T_ci is derived, per the source script's own behaviour.
    """
    power_kw: float       # Power_kW - compressor electrical power
    qevap_kw: float       # Q_evap_kW - evaporator cooling rate
    t_evap_in_c: float    # T_ei - chilled water / evaporator INLET temperature (degC)
    t_evap_out_c: float   # T_eo - chilled water / evaporator OUTLET temperature (degC)
    dataset: str          # "sql" or "rp1043"
    t_cond_in_c: float = None  # T_ci - condenser water INLET temperature (degC).
                                 # Only used for dataset="rp1043"; ignored (and
                                 # replaced by t_evap_in_c) for dataset="sql",
                                 # exactly as the training script does.


def _to_kelvin(celsius: float) -> float:
    return celsius + 273.15


def _derive_kelvin_temps(r: RawReading):
    t_ei_k = _to_kelvin(r.t_evap_in_c)
    t_eo_k = _to_kelvin(r.t_evap_out_c)
    if r.dataset == "rp1043":
        if r.t_cond_in_c is None:
            raise ValueError("t_cond_in_c is required for dataset='rp1043'")
        t_ci_k = _to_kelvin(r.t_cond_in_c)
    else:
        # SQL/EnergyPlus: original script sets T_ci_K = T_ei_K (no real
        # condenser-side variable was extracted from the SQL file).
        t_ci_k = t_ei_k
    return t_ei_k, t_eo_k, t_ci_k


def gordon_ng_inputs(r: RawReading):
    """x1, x2 fed to the Gordon-Ng LinearRegression, exactly as
    train_physics()/add_residuals() compute them in the source script."""
    t_ei_k, t_eo_k, t_ci_k = _derive_kelvin_temps(r)
    q = max(r.qevap_kw, EPS)
    x1 = t_ci_k / q
    x2 = t_eo_k / q
    return x1, x2


def compute_features(r: RawReading, physics_model) -> dict:
    """Returns the 12 named features, in the same units/definitions as
    compute_thermo() + add_residuals() in the training script, BEFORE
    StandardScaler.transform()."""

    t_ei_k, t_eo_k, t_ci_k = _derive_kelvin_temps(r)
    q = max(r.qevap_kw, EPS)
    p = r.power_kw

    # --- compute_thermo() ---
    cop = r.qevap_kw / max(p, EPS)
    power_per_load = p / q
    delta_t_lift = t_ci_k - t_eo_k
    delta_t_norm = delta_t_lift / q
    dT = max(t_ci_k - t_eo_k, EPS)
    evap_effectiveness = r.qevap_kw / dT
    q_cond = r.qevap_kw + p
    cond_dt = q_cond / q
    heat_rejection_ratio = q_cond / q
    comp_work_ratio = p / max(q_cond, EPS)
    evap_approach = (t_ei_k - t_eo_k)  # T_ei_K is always populated here (required input)
    cop_c = t_eo_k / max(t_ci_k - t_eo_k, EPS)
    cop_ratio = cop / max(cop_c, EPS)

    # --- add_residuals() ---
    x1, x2 = t_ci_k / q, t_eo_k / q
    pred_ratio = physics_model.predict([[x1, x2]])[0]  # predicts Power/Qevap
    pred_power = r.qevap_kw * pred_ratio
    power_residual = p - pred_power
    power_residual_norm = power_residual / q

    return {
        "Power_Residual": power_residual,
        "Power_Residual_norm": power_residual_norm,
        "COP": cop,
        "Power_per_load": power_per_load,
        "delta_T_lift": delta_t_lift,
        "delta_T_norm": delta_t_norm,
        "Evap_effectiveness": evap_effectiveness,
        "COP_ratio": cop_ratio,
        "Heat_rejection_ratio": heat_rejection_ratio,
        "Comp_work_ratio": comp_work_ratio,
        "Cond_dT": cond_dt,
        "Evap_approach": evap_approach,
    }


def feature_vector(r: RawReading, physics_model):
    feats = compute_features(r, physics_model)
    return [feats[name] for name in FEATURE_ORDER]
