"""Königs Wusterhausen municipal feature builder (shared by all validators).

Builds the *exact same* scale-free feature contract used in training
(src.features.schema.FEATURE_COLS): local physical priors expressed as capacity
factors (prior_mw / local_nameplate_mw), spatial-mean weather, cyclical time.
Also returns the E.ON telemetry actuals (kW, UTC) and local nameplate capacities.

Scrape timestamps are UTC (verified empirically: PV/wind cross-correlate with
UTC weather at 0-1 h lag, not the +2 h a Berlin-local reading would imply), so
no timezone conversion is applied.
"""

import os
import logging
import numpy as np
import pandas as pd

from src.models.wind_physics_transformer import WindPowerCalculator
from src.features.feature_pipeline import SOLAR_PERFORMANCE_RATIO
from src.features.schema import FEATURE_COLS
from src.data.boundary_fetcher import load_config

logger = logging.getLogger(__name__)

KW_LAT, KW_LON = 52.298, 13.626
RADIUS_KM = 8.0


def load_municipalities(config_path='conf/config.yaml'):
    """Configured municipal validation targets (dicts: slug, lat, lon, telemetry, results_dir)."""
    cfg = load_config(config_path)
    out = []
    for slug, r in cfg['scraper']['regions'].items():
        if 'lat' not in r or 'lon' not in r:
            logger.warning(f"region {slug} has no lat/lon; skipping")
            continue
        out.append({'slug': slug, 'lat': float(r['lat']), 'lon': float(r['lon']),
                    'telemetry': r['csv_path'],
                    'results_dir': r.get('results_dir', f"results/{slug}")})
    return out


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    dlat = p2 - p1
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def load_edis_telemetry(path='data/webscrape_kwusterhausen.csv'):
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)  # scrape is UTC
    df = df.drop_duplicates(subset=['timestamp']).set_index('timestamp').sort_index()
    return df[['photovoltaik', 'windkraft']].resample('1h').mean()


def municipal_fleet(lat, lon, radius_km=RADIUS_KM):
    """Local wind/solar MaStR assets within radius_km, their nameplate MW, and the
    unique 0.1-snapped weather-node list. Shared by the archive validator and the
    live-forecast dashboard so both build the identical scale-free feature."""
    wind_mastr = pd.read_csv('data/raw/mastr_wind_brandenburg_raw.csv')
    solar_mastr = pd.read_csv('data/raw/mastr_solar_brandenburg_raw.csv')
    wind_mastr['distance'] = haversine_distance(wind_mastr['latitude'], wind_mastr['longitude'], lat, lon)
    solar_mastr['distance'] = haversine_distance(solar_mastr['latitude'], solar_mastr['longitude'], lat, lon)
    kw_wind = wind_mastr[wind_mastr['distance'] <= radius_km].copy()
    kw_solar = solar_mastr[solar_mastr['distance'] <= radius_km].copy()

    wind_nameplate_mw = kw_wind['gross_power'].sum() / 1000.0
    solar_nameplate_mw = kw_solar['gross_power'].sum() / 1000.0
    logger.info(f"Fleet @({lat},{lon}) r={radius_km}km: wind {wind_nameplate_mw:.2f} MW ({len(kw_wind)}), solar {solar_nameplate_mw:.2f} MW ({len(kw_solar)})")

    for d in (kw_wind, kw_solar):
        d['lat_snap'] = d['latitude'].round(1)
        d['lon_snap'] = d['longitude'].round(1)
    nodes = pd.concat([kw_wind[['lat_snap', 'lon_snap']], kw_solar[['lat_snap', 'lon_snap']]]).drop_duplicates()
    return kw_wind, kw_solar, wind_nameplate_mw, solar_nameplate_mw, nodes


def assemble_municipal_features(kw_wind, kw_solar, wind_nameplate_mw, solar_nameplate_mw, nodes, load_node_weather):
    """Build the scale-free FEATURE_COLS matrix from per-node weather.

    `load_node_weather(lat, lon)` returns a UTC-indexed DataFrame with the weather
    columns (or None if unavailable). Identical physics whether that weather comes
    from the stored archive parquet (validation) or the live forecast API (dashboard)
    — this is what keeps the persisted affine calibration valid at inference time.
    Returns (X[FEATURE_COLS], caps_mw dict).
    """
    calc = WindPowerCalculator()
    weather_dfs, wind_priors, solar_priors = [], [], []
    for _, node in nodes.iterrows():
        lat, lon = node['lat_snap'], node['lon_snap']
        w = load_node_weather(lat, lon)
        if w is None:
            logger.warning(f"missing weather node {lat},{lon}")
            continue
        weather_dfs.append(w)

        nw = kw_wind[(kw_wind['lat_snap'] == lat) & (kw_wind['lon_snap'] == lon)]
        if not nw.empty:
            v = w['wind_speed_100m'].values / 3.6
            p = np.zeros(len(w))
            for _, a in nw.iterrows():
                try:
                    p = p + calc.calculate_power(
                        v_100m=v, hub_height=a['hub_height'] if pd.notna(a['hub_height']) else 100.0,
                        manufacturer_id=None, type_designation="Generic",
                        gross_power=a['gross_power'],
                        rotor_diameter=a['rotor_diameter'] if pd.notna(a['rotor_diameter']) else None)
                except Exception:
                    pass
            wind_priors.append(pd.DataFrame({'p': p / 1000.0}, index=w.index))

        ns = kw_solar[(kw_solar['lat_snap'] == lat) & (kw_solar['lon_snap'] == lon)]
        if not ns.empty:
            cap_mw = ns['gross_power'].sum() / 1000.0
            # Same STC-normalised physics as training (the old validator omitted /1000).
            pw = (w['shortwave_radiation'] / 1000.0) * cap_mw * SOLAR_PERFORMANCE_RATIO
            solar_priors.append(pd.DataFrame({'p': pw}, index=w.index))

    if not weather_dfs:
        raise RuntimeError(
            "No weather data for any node of this fleet — run the weather "
            "ingestion/extension first (see missing-node warnings above).")
    idx0 = weather_dfs[0].index
    wind_prior_mw = pd.concat(wind_priors, axis=1).sum(axis=1) if wind_priors else pd.Series(0.0, index=idx0)
    solar_prior_mw = pd.concat(solar_priors, axis=1).sum(axis=1) if solar_priors else pd.Series(0.0, index=idx0)

    combined = pd.concat(weather_dfs)
    regional_mean = combined.groupby(combined.index).mean()

    X = pd.DataFrame(index=regional_mean.index)
    # Priors as capacity factors against LOCAL nameplate -> scale-free, matches training.
    X['wind_prior_cf'] = (wind_prior_mw / wind_nameplate_mw).clip(0, 1.5) if wind_nameplate_mw > 0 else 0.0
    X['solar_prior_cf'] = (solar_prior_mw / solar_nameplate_mw).clip(0, 1.5) if solar_nameplate_mw > 0 else 0.0
    for c in ['temperature_2m', 'surface_pressure', 'relative_humidity_2m']:
        X[c] = regional_mean[c]
    h, m = X.index.hour, X.index.month
    X['hour_sin'] = np.sin(2 * np.pi * h / 24.0)
    X['hour_cos'] = np.cos(2 * np.pi * h / 24.0)
    X['month_sin'] = np.sin(2 * np.pi * m / 12.0)
    X['month_cos'] = np.cos(2 * np.pi * m / 12.0)

    X = X[FEATURE_COLS].dropna().sort_index()
    return X, {'wind': wind_nameplate_mw, 'solar': solar_nameplate_mw}


def build_municipal_features(lat, lon, telemetry_path, radius_km=RADIUS_KM):
    """Scale-free feature builder for any municipality (same contract as training),
    reading the stored archive weather nodes. Returns
    (X[FEATURE_COLS], actuals[photovoltaik,windkraft] kW, caps_mw dict)."""
    actuals = load_edis_telemetry(telemetry_path)
    fleet = municipal_fleet(lat, lon, radius_km)

    def load_node(la, lo):
        f = f"data/processed/weather/grid_nodes/node_{la}_{lo}.parquet"
        if not os.path.exists(f):
            return None
        w = pd.read_parquet(f)
        w['time'] = pd.to_datetime(w['time'], utc=True)
        return w.set_index('time')

    X, caps = assemble_municipal_features(*fleet, load_node)
    common = X.index.intersection(actuals.index)
    return X.loc[common], actuals.loc[common], caps


def build_kw_features():
    """Backward-compat wrapper: Königs Wusterhausen with default radius."""
    return build_municipal_features(KW_LAT, KW_LON, 'data/webscrape_kwusterhausen.csv')


def calibrate_affine(y_true_kw, base_pred_kw):
    """2-parameter local bias corrector: fit Local = scale * base + offset on the
    calibration split, returning (scale, offset).

    Fit by ordinary least squares *with an intercept*. Including the intercept
    makes the mean residual identically zero on the fit split, i.e. MBE == 0 by
    construction — this is the requested "push the MBE to zero" — while OLS keeps
    the problem well-posed. (Minimising MBE alone is degenerate: scale=0,
    offset=mean(y) also gives MBE=0 but discards all skill; OLS picks the unique
    best-fit line among the MBE=0 solutions.)

    Falls back to offset-only (scale=1) when the base is degenerate/constant or
    OLS returns a non-positive slope, so the MBE is still zeroed sensibly.
    """
    a = np.asarray(y_true_kw, float)
    b = np.asarray(base_pred_kw, float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 2 or np.std(b) < 1e-9:
        return 1.0, (float(np.mean(a - b)) if len(a) else 0.0)
    scale, offset = np.polyfit(b, a, 1)  # deg-1 OLS -> [slope, intercept]
    if not np.isfinite(scale) or scale <= 0:
        return 1.0, float(np.mean(a - b))
    return float(scale), float(offset)


def apply_affine(base_pred_kw, scale, offset):
    """Municipal bias-corrected generation from the uncalibrated CF*cap base.
    Physical floor at zero (generation is non-negative)."""
    return (np.asarray(base_pred_kw, float) * scale + offset).clip(min=0.0)


if __name__ == '__main__':
    # Self-check: the intercept fit recovers a known affine relation and drives
    # the fit-split MBE to (numerical) zero; the degenerate-base path still zeroes it.
    rng = np.random.default_rng(0)
    b = rng.uniform(0, 5000, 500)
    a = 2.0 * b + 300.0 + rng.normal(0, 50, 500)  # true scale=2, offset=300
    s, o = calibrate_affine(a, b)
    assert abs(s - 2.0) < 0.05 and abs(o - 300.0) < 40, (s, o)
    mbe = float(np.mean(apply_affine(b, s, o) - a))  # all preds positive -> clip is a no-op
    assert abs(mbe) < 1.0, f"fit-split MBE should be ~0, got {mbe}"

    s2, o2 = calibrate_affine(a, np.full_like(b, 7.0))  # constant base -> offset-only
    assert s2 == 1.0 and abs(np.mean((np.full_like(b, 7.0) + o2) - a)) < 1e-6
    print(f"calibrate_affine self-check passed: scale={s:.3f} offset={o:.1f}")
