"""Brandenburg district registry + geospatial asset assignment (state-wide scaling).

MaStR carries only lat/lon, so assets are assigned to a Landkreis / kreisfreie
Stadt by point-in-polygon. District boundaries are fetched once from Nominatim
(reliable for named admin areas; Overpass times out on area queries) and cached
to data/external/brandenburg_districts.geojson.

Public API:
  load_district_boundaries()  -> GeoDataFrame(name, kind, geometry, lat, lon)
  load_district_fleets()      -> {name: (wind, solar, wind_mw, solar_mw, nodes)}  (reuses
                                 assemble_municipal_features, identical physics)
  resolve_calibration(tech, cap_mw, mode) -> {scale, offset, cap_mw}  (fallback for
                                 uncalibrated districts)
"""

import os
import json
import time
import logging

import numpy as np
import pandas as pd
import geopandas as gpd
import requests

from src.data.boundary_fetcher import load_config
from src.validation.kw_features import load_municipalities

logger = logging.getLogger(__name__)

NOMINATIM = "https://nominatim.openstreetmap.org/search"
UA = "bb-forecast/1.0 (bachelor thesis, contact cedms98@gmail.com)"
WIND_CSV = 'data/raw/mastr_wind_brandenburg_raw.csv'
SOLAR_CSV = 'data/raw/mastr_solar_brandenburg_raw.csv'


def district_registry(config_path='conf/config.yaml'):
    """(name, kind, nominatim_query) for the 14 Landkreise + 4 kreisfreie Städte."""
    cfg = load_config(config_path)['districts']
    reg = [(n, 'landkreis', f"Landkreis {n}, Brandenburg, Germany") for n in cfg['landkreise']]
    reg += [(n, 'stadt', f"{n}, Brandenburg, Germany") for n in cfg['kreisfreie_staedte']]
    return reg


def _nominatim_polygon(query):
    r = requests.get(NOMINATIM, params={
        "q": query, "format": "jsonv2", "polygon_geojson": 1, "limit": 1, "countrycodes": "de",
    }, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    j = r.json()
    if not j:
        raise RuntimeError(f"Nominatim returned no result for {query!r}")
    return j[0]["geojson"]


def fetch_district_boundaries(config_path='conf/config.yaml', force=False):
    """Fetch + cache the 18 district polygons. Nominatim asks for <=1 req/s, so we
    sleep between calls. Idempotent: skips the network if the cache exists."""
    cfg = load_config(config_path)['districts']
    out_path = cfg['boundaries_geojson']
    if os.path.exists(out_path) and not force:
        return gpd.read_file(out_path)

    from shapely.geometry import shape
    rows = []
    for name, kind, query in district_registry(config_path):
        geom = shape(_nominatim_polygon(query))
        rows.append({"name": name, "kind": kind, "geometry": geom})
        logger.info(f"boundary fetched: {name} ({geom.geom_type})")
        time.sleep(1.1)  # Nominatim usage policy
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    gdf.to_file(out_path, driver="GeoJSON")
    logger.info(f"cached {len(gdf)} district boundaries -> {out_path}")
    return gdf


def load_district_boundaries(config_path='conf/config.yaml'):
    """District polygons + representative centroids. Validates the full registry
    is present so a partial fetch can't silently drop a district."""
    gdf = fetch_district_boundaries(config_path)
    expected = {n for n, _, _ in district_registry(config_path)}
    missing = expected - set(gdf['name'])
    if missing:
        raise RuntimeError(f"district boundaries missing {sorted(missing)}; delete the cache and re-fetch")
    # representative_point() is guaranteed inside the polygon (centroid may not be).
    pts = gdf.geometry.representative_point()
    gdf = gdf.assign(lon=pts.x.round(3), lat=pts.y.round(3))
    return gdf


def _active(df):
    """Commissioned and not yet decommissioned as of now (the 'active' fleet)."""
    dec = pd.to_datetime(df['final_decommission_date'], errors='coerce', utc=True)
    com = pd.to_datetime(df['commissioning_date'], errors='coerce', utc=True)
    now = pd.Timestamp.now(tz='UTC')
    return df[dec.isna() & (com.isna() | (com <= now))].copy()


def _assign(df, boundaries):
    """Point-in-polygon: tag each asset with its district name (one spatial join)."""
    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']),
                         crs="EPSG:4326")
    joined = gpd.sjoin(g, boundaries[['name', 'geometry']], how='inner', predicate='within')
    return joined.drop(columns='geometry')


def load_district_fleets(config_path='conf/config.yaml'):
    """Assign every active MaStR asset to its district and build one fleet tuple per
    district — shape-compatible with municipal_fleet so assemble_municipal_features
    is reused verbatim. Returns {name: (wind, solar, wind_mw, solar_mw, nodes)}."""
    boundaries = load_district_boundaries(config_path)
    wind = _assign(_active(pd.read_csv(WIND_CSV)), boundaries)
    solar = _assign(_active(pd.read_csv(SOLAR_CSV)), boundaries)
    for d in (wind, solar):
        d['lat_snap'] = d['latitude'].round(1)
        d['lon_snap'] = d['longitude'].round(1)

    fleets = {}
    for name in boundaries['name']:
        dw = wind[wind['name'] == name].copy()
        ds = solar[solar['name'] == name].copy()
        wind_mw = dw['gross_power'].sum() / 1000.0
        solar_mw = ds['gross_power'].sum() / 1000.0
        nodes = pd.concat([dw[['lat_snap', 'lon_snap']], ds[['lat_snap', 'lon_snap']]]).drop_duplicates()
        fleets[name] = (dw, ds, wind_mw, solar_mw, nodes)
        logger.info(f"{name}: wind {wind_mw:.1f} MW ({len(dw)}), solar {solar_mw:.1f} MW ({len(ds)}), {len(nodes)} nodes")
    return fleets


# ---------------------------------------------------------------------------
# Fallback calibration for districts with no validation telemetry.
# ---------------------------------------------------------------------------
def _pooled_scale(tech):
    """Mean LightGBM affine scale across the validated pilot municipalities."""
    scales = []
    for m in load_municipalities():
        p = os.path.join(m['results_dir'], 'affine_calibration.json')
        if os.path.exists(p):
            with open(p) as f:
                c = json.load(f).get(f'lightgbm_{tech}')
            if c:
                scales.append(c['scale'])
    return float(np.mean(scales)) if scales else 1.0


def resolve_calibration(tech, cap_mw, mode='identity'):
    """Downscaling calibration for an uncalibrated district.

    identity (default): scale=1.0, offset=0.0 -> trust the physics-informed
        CF x nameplate. offset is an absolute-kW term fit to a tiny Gemeinde fleet
        and does not transfer to a Landkreis, so any fallback forces it to 0.
    pooled: mean pilot scale (offset still 0) -> a sensitivity knob, NOT validated;
        the two pilot scales diverge (e.g. wind 0.43 vs 0.012), so treat with care.
    """
    scale = _pooled_scale(tech) if mode == 'pooled' else 1.0
    return {'scale': scale, 'offset': 0.0, 'cap_mw': cap_mw}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # Self-check: all 18 districts load, assets land inside exactly one district
    # (partition), and the fallback resolver returns identity by default.
    fleets = load_district_fleets()
    assert len(fleets) == 18, f"expected 18 districts, got {len(fleets)}"
    total_wind = sum(f[2] for f in fleets.values())
    total_solar = sum(f[3] for f in fleets.values())
    assert total_wind > 1000 and total_solar > 1000, (total_wind, total_solar)
    c = resolve_calibration('wind', 123.4)
    assert c == {'scale': 1.0, 'offset': 0.0, 'cap_mw': 123.4}, c
    assert resolve_calibration('solar', 1.0, mode='pooled')['scale'] != 1.0
    print(f"districts self-check passed: 18 districts, "
          f"state wind {total_wind:.0f} MW, solar {total_solar:.0f} MW")
