"""Live 24h-ahead municipal generation forecast (backend orchestration for the UI).

On demand for a chosen municipality:
  1. fetch live Open-Meteo *forecast* weather at the local weather nodes,
  2. build the identical scale-free features used in validation,
  3. run the LightGBM quantile models (q10/q50/q90) -> capacity factor,
  4. downscale with the municipality's persisted affine calibration + nameplate
     (Local = scale * (CF * cap_mw * 1000) + offset), same q10/q50/q90 band.

Output is MW. The median is the point forecast; [q10, q90] is the 80% interval.
"""

import os
import json
import logging

import numpy as np
import pandas as pd
import requests
import joblib

from src.data.boundary_fetcher import load_config
from src.validation.kw_features import (
    municipal_fleet, assemble_municipal_features, apply_affine, load_municipalities,
)
from src.models.train_lightgbm import QUANTILES, quantile_model_path
from src.features.schema import FEATURE_COLS, CF_CLIP

logger = logging.getLogger(__name__)

# Weather variables the feature builder / physics priors actually consume.
FORECAST_VARS = [
    "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "wind_speed_100m", "shortwave_radiation",
]
CALIB_NAME = 'affine_calibration.json'


def local_day_window(tz="Europe/Berlin"):
    """UTC (start, end) bounding [00:00, 24:00) of the current calendar day in `tz`."""
    start_local = pd.Timestamp.now(tz=tz).normalize()
    return start_local.tz_convert('UTC'), (start_local + pd.Timedelta(days=1)).tz_convert('UTC')


def fetch_forecast_weather(node_coords, forecast_days=2, past_days=1, max_retries=4, pause=0.0):
    """Open-Meteo forecast weather for each (lat, lon) node -> {(lat,lon): DataFrame}.
    One batched API call for all nodes, with exponential backoff on 429/5xx so the
    sequential state-wide batch loop degrades gracefully under rate-limiting.
    `past_days=1` guarantees the response covers local midnight even when it falls
    before the UTC day boundary (CEST is UTC+2), so callers slicing to a local
    calendar-day window always have full coverage. `pause` sleeps after a
    successful call (politeness between districts)."""
    import time
    cfg = load_config()
    url = cfg['open_meteo']['forecast_url']
    lats = ",".join(str(la) for la, lo in node_coords)
    lons = ",".join(str(lo) for la, lo in node_coords)
    params = {"latitude": lats, "longitude": lons, "hourly": ",".join(FORECAST_VARS),
              "forecast_days": forecast_days, "past_days": past_days, "timezone": "UTC"}

    backoff = 2.0
    for attempt in range(max_retries):
        r = requests.get(url, params=params, timeout=40)
        if r.status_code == 429 or r.status_code >= 500:
            if attempt == max_retries - 1:
                r.raise_for_status()
            logger.warning(f"Open-Meteo {r.status_code}; backing off {backoff:.0f}s "
                           f"(attempt {attempt + 1}/{max_retries})")
            time.sleep(backoff)
            backoff *= 2
            continue
        r.raise_for_status()
        break

    data = r.json()
    if isinstance(data, dict):  # single-location responses are not wrapped in a list
        data = [data]
    out = {}
    for (la, lo), d in zip(node_coords, data):
        h = d.get("hourly", {})
        if not h.get("time"):
            continue
        df = pd.DataFrame(h)
        df['time'] = pd.to_datetime(df['time'], utc=True)
        out[(la, lo)] = df.set_index('time')
    if pause:
        time.sleep(pause)
    return out


def _downscale_interval(X, tech, c):
    """LightGBM quantile CF -> calibrated MW (lower, median, upper) for `tech`.
    `c` is the resolved per-tech calibration {scale, offset, cap_mw}."""
    P = np.vstack([np.clip(joblib.load(quantile_model_path(tech, q)).predict(X[FEATURE_COLS]), 0.0, CF_CLIP)
                   for q in QUANTILES]).T
    P.sort(axis=1)  # enforce q10 <= q50 <= q90
    cols = {}
    for label, col in (('lower', 0), ('median', 1), ('upper', 2)):
        kw = apply_affine(P[:, col] * c['cap_mw'] * 1000.0, c['scale'], c['offset'])
        cols[f'{tech}_{label}_mw'] = kw / 1000.0  # kW -> MW
    return pd.DataFrame(cols, index=X.index)


def forecast_from_fleet(fleet, calib_provider, horizon_h=24, weather=None, pause=0.0, window=None):
    """Core inference shared by the municipal (radius) and district (polygon) paths.

    `fleet` is the (wind, solar, wind_mw, solar_mw, nodes) tuple; `calib_provider(tech,
    cap_mw)` returns the resolved {scale, offset, cap_mw}. `window`, if given, is a
    (start_utc, end_utc) tuple (see `local_day_window`) selecting a fixed calendar
    window instead of the default rolling next-`horizon_h` from now. Returns
    (forecast_df, caps_mw).
    """
    nodes = fleet[4]
    node_coords = list(dict.fromkeys(zip(nodes['lat_snap'], nodes['lon_snap'])))
    if weather is None:
        weather = fetch_forecast_weather(node_coords, pause=pause)

    X, caps = assemble_municipal_features(*fleet, lambda la, lo: weather.get((la, lo)))
    if window is not None:
        start, end = window
        X = X[(X.index >= start) & (X.index < end)]
    else:
        now = pd.Timestamp.now(tz='UTC').ceil('h')
        X = X[X.index >= now].head(horizon_h)
    if X.empty:
        raise RuntimeError("No forecast hours in requested window; weather fetch may have failed.")

    out = pd.concat([_downscale_interval(X, 'wind', calib_provider('wind', caps['wind'])),
                     _downscale_interval(X, 'solar', calib_provider('solar', caps['solar']))], axis=1)
    return out, caps


def forecast_municipality(muni, horizon_h=24, weather=None):
    """24h-ahead wind+solar MW interval forecast for a calibrated pilot municipality.
    Returns (forecast_df, caps_mw, calib) — the fitted affine calibration."""
    with open(os.path.join(muni['results_dir'], CALIB_NAME)) as f:
        calib = json.load(f)
    fleet = municipal_fleet(muni['lat'], muni['lon'])
    out, caps = forecast_from_fleet(fleet, lambda tech, cap: calib[f'lightgbm_{tech}'],
                                    horizon_h, weather)
    return out, caps, calib


def forecast_district(name, fleet, mode='identity', horizon_h=24, weather=None, pause=0.0, window=None):
    """24h-ahead forecast for an (uncalibrated) district using the fallback calibration.
    Returns (forecast_df, caps_mw, calib_used)."""
    from src.data.districts import resolve_calibration
    used = {}

    def provider(tech, cap):
        used[tech] = resolve_calibration(tech, cap, mode)
        return used[tech]

    out, caps = forecast_from_fleet(fleet, provider, horizon_h, weather, pause=pause, window=window)
    return out, caps, used


def municipality_display_name(slug):
    return {'koenigs_wusterhausen': 'Königs Wusterhausen', 'nauen': 'Nauen'}.get(
        slug, slug.replace('_', ' ').title())


if __name__ == "__main__":
    # Network-free self-check: feed synthetic forecast weather for KW's nodes and
    # assert the interval is ordered (lower<=median<=upper) and physically valid (MW>=0).
    logging.basicConfig(level=logging.WARNING)
    kw = next(m for m in load_municipalities() if m['slug'] == 'koenigs_wusterhausen')
    fleet = municipal_fleet(kw['lat'], kw['lon'])
    coords = list(dict.fromkeys(zip(fleet[4]['lat_snap'], fleet[4]['lon_snap'])))
    idx = pd.date_range(pd.Timestamp.now(tz='UTC').ceil('h'), periods=48, freq='h')
    rng = np.random.default_rng(0)
    fake = {c: pd.DataFrame({
        'temperature_2m': 15 + 5 * rng.random(len(idx)),
        'relative_humidity_2m': 60 + 20 * rng.random(len(idx)),
        'surface_pressure': 1010 + 5 * rng.random(len(idx)),
        'wind_speed_100m': 20 + 15 * rng.random(len(idx)),          # km/h
        'shortwave_radiation': np.clip(600 * np.sin(np.pi * idx.hour / 24.0), 0, None),
    }, index=idx) for c in coords}

    fc, caps, calib = forecast_municipality(kw, horizon_h=24, weather=fake)
    assert len(fc) == 24, f"expected 24 rows, got {len(fc)}"
    for tech in ('wind', 'solar'):
        lo, med, hi = fc[f'{tech}_lower_mw'], fc[f'{tech}_median_mw'], fc[f'{tech}_upper_mw']
        assert (lo <= med + 1e-9).all() and (med <= hi + 1e-9).all(), f"{tech} band not ordered"
        assert (lo >= -1e-9).all(), f"{tech} negative MW"
    print("forecast_service self-check passed:", {t: round(fc[f'{t}_median_mw'].max(), 2) for t in ('wind', 'solar')})
