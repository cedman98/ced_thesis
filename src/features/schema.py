"""Single source of truth for the ML matrix column contract.

Centralised so every trainer and validator selects the *same* scale-free feature
set and never leaks a target / capacity / raw-MW column into X.
"""

# Scale-free features. Priors are expressed as capacity factors (dimensionless,
# ~[0, 1]) so the learned map transfers unchanged from the macro (50Hertz zone)
# scale to a single municipality. Weather intensities and cyclical time are
# already scale-free.
FEATURE_COLS = [
    "wind_prior_cf",
    "solar_prior_cf",
    "temperature_2m",
    "surface_pressure",
    "relative_humidity_2m",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]

# Targets are capacity factors, not absolute MW (this is what CLAUDE.md always
# described; it was never actually implemented until now).
TARGET_CF = {"wind": "wind_cf_50hz", "solar": "solar_cf_50hz"}

# CF predictions/targets are clipped to [0, CF_CLIP] everywhere. >1 is legal
# (the effective-capacity proxy can undershoot true nameplate); trainers and
# validators must use the SAME bound or municipal results silently diverge.
CF_CLIP = 1.5

# Raw MW targets + the effective-capacity series used to build the CF, kept in
# the matrix so MW can be reconstructed for reporting. Never fed to a model.
TARGET_MW = {"wind": "wind_onshore_mw_50hz", "solar": "solar_pv_mw_50hz"}
CAP_COLS = {"wind": "wind_eff_capacity_mw", "solar": "solar_eff_capacity_mw"}

# The TFT decoder needs the prior + weather as known-future reals (same list,
# minus the two priors it splits by technology).
def tft_known_reals(tech: str) -> list:
    prior = "wind_prior_cf" if tech == "wind" else "solar_prior_cf"
    return [prior] + FEATURE_COLS[2:]
