"""
mastr_downloader.py

Automated download utility for the Energy Forecast System. Handles:
1. Streaming download and selective extraction of the Marktstammdatenregister (MaStR) bulk XML files.
2. Programmatic querying of the Overpass API for the Brandenburg administrative boundary GeoJSON.
3. Fetching of German postal code coordinates database.
"""

import os
import logging
import zipfile
import shutil
from pathlib import Path
from typing import Dict, Any
import requests
import yaml
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import LineString
from shapely.ops import linemerge, polygonize, unary_union

logger = logging.getLogger(__name__)

def load_config(config_path: str = "config/conf.yaml") -> Dict[str, Any]:
    """Loads the YAML configuration file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        A dictionary containing the configuration keys and values.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def download_mastr_bulk(config: Dict[str, Any]) -> None:
    """Stream-downloads the MaStR bulk ZIP archive and extracts target XML files.

    Args:
        config: The configuration dictionary.
    """
    mastr_conf = config["marktstammdatenregister"]
    url = mastr_conf["bulk_download_url"]
    
    wind_file = mastr_conf["wind"]["file_name"]
    wind_local = Path(mastr_conf["wind"]["local_path"])
    
    solar_file = mastr_conf["solar"]["file_name"]
    solar_local = Path(mastr_conf["solar"]["local_path"])
    
    # If the final XML files already exist, skip download to save bandwidth and time
    if wind_local.exists() and solar_local.exists():
        logger.info("MaStR XML files already exist locally. Skipping bulk download.")
        return

    # Ensure parent directories exist
    wind_local.parent.mkdir(parents=True, exist_ok=True)
    solar_local.parent.mkdir(parents=True, exist_ok=True)
    
    temp_zip_path = wind_local.parent / "temp_mastr_export.zip"
    
    logger.info(f"Starting streaming download of MaStR bulk archive from: {url}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get("content-length", 0))
        chunk_size = 1024 * 1024  # 1 MB chunk size
        
        with open(temp_zip_path, "wb") as f, tqdm(
            desc="Downloading MaStR ZIP",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
                    
        logger.info("Download completed. Extracting target XML files...")
        
        with zipfile.ZipFile(temp_zip_path, "r") as zf:
            extracted_wind = False
            extracted_solar = False
            
            for member in zf.namelist():
                member_name = member.split("/")[-1]
                if member_name == wind_file:
                    logger.info(f"Extracting {member} to {wind_local}")
                    with zf.open(member) as source, open(wind_local, "wb") as target:
                        shutil.copyfileobj(source, target)
                    extracted_wind = True
                elif member_name == solar_file:
                    logger.info(f"Extracting {member} to {solar_local}")
                    with zf.open(member) as source, open(solar_local, "wb") as target:
                        shutil.copyfileobj(source, target)
                    extracted_solar = True
                    
            if not extracted_wind:
                logger.warning(f"Could not find {wind_file} in the ZIP archive.")
            if not extracted_solar:
                logger.warning(f"Could not find {solar_file} in the ZIP archive.")
                
    except Exception as e:
        logger.error(f"Failed during MaStR download/extraction: {e}")
        raise
    finally:
        if temp_zip_path.exists():
            logger.info(f"Purging temporary ZIP archive: {temp_zip_path}")
            temp_zip_path.unlink()

def download_brandenburg_boundary(config: Dict[str, Any]) -> None:
    """Queries Overpass API to fetch the administrative boundary of Brandenburg and saves as GeoJSON.

    Args:
        config: The configuration dictionary.
    """
    osm_conf = config["open_meteo"]["openstreetmap"]
    relation_id = osm_conf["relation_id"]
    api_url = osm_conf["overpass_api_url"]
    geojson_path = Path(osm_conf["geojson_local_path"])
    
    if geojson_path.exists():
        logger.info(f"Brandenburg boundary GeoJSON already exists at: {geojson_path}")
        return
        
    geojson_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Querying Overpass API for Brandenburg relation ID {relation_id}")
    query = f"[out:json];relation({relation_id});out geom;"
    headers = {
        "User-Agent": "BrandenburgEnergyForecastThesis/1.0 (contact: student@bachelor-thesis-project.de)"
    }
    
    try:
        response = requests.post(api_url, data={"data": query}, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        elements = data.get("elements", [])
        if not elements:
            raise ValueError("No elements returned in Overpass response.")
            
        relation = elements[0]
        outer_lines = []
        inner_lines = []
        
        for member in relation.get("members", []):
            if member.get("type") == "way" and "geometry" in member:
                coords = [(pt["lon"], pt["lat"]) for pt in member["geometry"]]
                if len(coords) >= 2:
                    line = LineString(coords)
                    if member.get("role") == "inner":
                        inner_lines.append(line)
                    else:
                        outer_lines.append(line)
                        
        if not outer_lines:
            raise ValueError("No outer boundary lines found for relation.")
            
        outer_merged = linemerge(outer_lines)
        outer_polys = list(polygonize(outer_merged))
        outer_geom = unary_union(outer_polys)
        
        if inner_lines:
            inner_merged = linemerge(inner_lines)
            inner_polys = list(polygonize(inner_merged))
            inner_geom = unary_union(inner_polys)
            brandenburg_geom = outer_geom.difference(inner_geom)
        else:
            brandenburg_geom = outer_geom
            
        gdf = gpd.GeoDataFrame(geometry=[brandenburg_geom], crs="EPSG:4326")
        gdf["relation_id"] = relation_id
        gdf["name"] = "Brandenburg"
        
        gdf.to_file(geojson_path, driver="GeoJSON")
        logger.info(f"Brandenburg boundary saved successfully to: {geojson_path}")
        
    except Exception as e:
        logger.error(f"Failed to fetch or build Brandenburg boundary: {e}")
        raise

def download_postal_codes_lookup() -> None:
    """Downloads the WZB German postal code lookup coordinates CSV if missing."""
    target_path = Path("data/external/de_postal_codes.csv")
    if target_path.exists():
        logger.info(f"Postal code coordinates lookup already exists at: {target_path}")
        return
        
    target_path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://raw.githubusercontent.com/WZBSocialScienceCenter/plz_geocoord/master/plz_geocoord.csv"
    
    logger.info(f"Downloading German postal code lookup database from: {url}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        logger.info(f"Postal codes lookup database saved to: {target_path}")
    except Exception as e:
        logger.error(f"Failed to download postal code database: {e}")
        raise

def run_all_downloads(config_path: str = "config/conf.yaml") -> None:
    """Orchestrates all download operations.

    Args:
        config_path: Path to the YAML configuration file.
    """
    config = load_config(config_path)
    download_mastr_bulk(config)
    download_brandenburg_boundary(config)
    download_postal_codes_lookup()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    run_all_downloads()
