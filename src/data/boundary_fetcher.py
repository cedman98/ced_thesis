"""Boundary Fetcher Utility.

This script fetches the official Brandenburg administrative boundary polygon (Relation ID: 62504)
from the OpenStreetMap Overpass API and saves it as a GeoJSON file for geographic masking.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict
import requests
import yaml
import geopandas as gpd
from shapely.geometry import LineString, Polygon
from shapely.ops import linemerge, polygonize, unary_union

logger = logging.getLogger(__name__)


def load_config(config_path: str = "conf/config.yaml") -> Dict[str, Any]:
    """Loads the YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A dictionary containing configuration parameters.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the configuration file contains invalid YAML syntax.
    """
    path = Path(config_path)
    if not path.exists():
        logger.error(f"Configuration file not found at: {config_path}")
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
            logger.info(f"Configuration loaded successfully from {config_path}")
            return config
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file: {e}")
            raise


def fetch_brandenburg_boundary(config: Dict[str, Any]) -> str:
    """Checks for the local GeoJSON boundary of Brandenburg.

    If missing, queries the Overpass API to construct the true administrative
    boundary polygon and saves it locally.

    Args:
        config: The loaded configuration dictionary.

    Returns:
        The string path to the saved/existing GeoJSON file.

    Raises:
        ValueError: If the API response contains no geometry or invalid structure.
        requests.RequestException: If the Overpass API request fails.
    """
    osm_conf = config["open_meteo"]["openstreetmap"]
    relation_id = osm_conf["relation_id"]
    api_url = osm_conf["overpass_api_url"]
    geojson_path = Path(osm_conf["geojson_local_path"])

    if geojson_path.exists():
        logger.info(f"Brandenburg boundary GeoJSON already exists at: {geojson_path}")
        return str(geojson_path)

    # Ensure output directory exists
    geojson_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Querying Overpass API for Brandenburg relation ID: {relation_id}")
    query = f"[out:json];relation({relation_id});out geom;"
    headers = {
        "User-Agent": "BrandenburgEnergyForecastThesis/1.0 (contact: student@bachelor-thesis-project.de)"
    }

    try:
        response = requests.post(api_url, data={"data": query}, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()

        elements = data.get("elements", [])
        if not elements:
            raise ValueError(
                f"No elements found in Overpass API response for relation {relation_id}."
            )

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
            raise ValueError("No outer boundary lines found for the Brandenburg relation.")

        # Reconstruct outer boundary polygon
        outer_merged = linemerge(outer_lines)
        outer_polys = list(polygonize(outer_merged))
        if not outer_polys:
            raise ValueError("Could not construct closed polygons from outer ways.")
        outer_geom = unary_union(outer_polys)

        # Reconstruct inner hole polygons (enclaves/exclusions) if any
        if inner_lines:
            inner_merged = linemerge(inner_lines)
            inner_polys = list(polygonize(inner_merged))
            inner_geom = unary_union(inner_polys)
            brandenburg_geom = outer_geom.difference(inner_geom)
        else:
            brandenburg_geom = outer_geom

        # Create GeoDataFrame and save
        gdf = gpd.GeoDataFrame(geometry=[brandenburg_geom], crs="EPSG:4326")
        gdf["relation_id"] = relation_id
        gdf["name"] = "Brandenburg"

        gdf.to_file(geojson_path, driver="GeoJSON")
        logger.info(f"Brandenburg boundary saved successfully to: {geojson_path}")
        return str(geojson_path)

    except requests.RequestException as e:
        logger.error(f"HTTP request to Overpass API failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to process and build Brandenburg boundary: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    conf = load_config()
    fetch_brandenburg_boundary(conf)
