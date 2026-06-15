"""Marktstammdatenregister Bulk XML Streaming Parser.

This module provides a memory-safe streaming parser for massive MaStR bulk XML data
using row-by-row parsing, coordinate imputation based on postal codes, and
spatial filtering using the Brandenburg administrative boundary GeoJSON.
"""

import os
import glob
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import xml.etree.ElementTree as ET
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ProgressFile:
    """File-like wrapper that reports read progress to a tqdm bar.

    Attributes:
        file: The underlying file object opened in binary mode.
        total: The total file size in bytes.
        bar: The tqdm progress bar instance.
    """

    def __init__(self, filepath: str, desc: str = "Reading") -> None:
        """Initializes ProgressFile wrapper.

        Args:
            filepath: Path to the file to open.
            desc: Description prefix for the tqdm progress bar.
        """
        self.file = open(filepath, "rb")
        self.total = os.path.getsize(filepath)
        self.bar = tqdm(
            total=self.total,
            desc=desc,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            leave=True,
        )

    def read(self, size: int = -1) -> bytes:
        """Reads data from the file and updates the progress bar.

        Args:
            size: The number of bytes to read. Defaults to -1 (read all).

        Returns:
            The bytes read from the file.
        """
        data = self.file.read(size)
        self.bar.update(len(data))
        return data

    def close(self) -> None:
        """Closes the file and progress bar."""
        self.bar.close()
        self.file.close()

    def __enter__(self) -> "ProgressFile":
        """Context management entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context management exit."""
        self.close()


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


def to_float(val: Optional[str]) -> Optional[float]:
    """Converts a string value to float, handling empty or invalid entries.

    Args:
        val: The string to convert.

    Returns:
        The float representation, or None if conversion failed.
    """
    if not val:
        return None
    try:
        return float(val.strip())
    except ValueError:
        return None


def get_brandenburg_centroid(geojson_path: str) -> Tuple[float, float]:
    """Retrieves the mathematical centroid of the Brandenburg boundary polygon.

    Args:
        geojson_path: Path to the GeoJSON boundary file.

    Returns:
        A tuple of (latitude, longitude) of the centroid.

    Raises:
        FileNotFoundError: If the GeoJSON file does not exist.
    """
    path = Path(geojson_path)
    if not path.exists():
        logger.error(f"Brandenburg boundary GeoJSON not found at: {geojson_path}")
        raise FileNotFoundError(f"Brandenburg boundary GeoJSON not found at: {geojson_path}")

    brandenburg_gdf = gpd.read_file(path)
    brandenburg_geom = brandenburg_gdf.geometry.iloc[0]
    centroid = brandenburg_geom.centroid
    logger.info(f"Calculated Brandenburg boundary centroid: Lat={centroid.y}, Lon={centroid.x}")
    return centroid.y, centroid.x


def parse_wind_units(
    xml_path: str,
    plz_df: pd.DataFrame,
    fallback_lat: float,
    fallback_lon: float,
) -> List[Dict[str, Any]]:
    """Memory-safely parses the wind XML file using iterparse.

    Args:
        xml_path: Path to the wind XML file.
        plz_df: German postal code lookup DataFrame.
        fallback_lat: Fallback latitude (centroid).
        fallback_lon: Fallback longitude (centroid).

    Returns:
        A list of parsed wind turbine records.
    """
    logger.info(f"Parsing wind units from: {xml_path}")
    records = []

    with ProgressFile(xml_path, desc="Parsing Wind XML") as f:
        context = ET.iterparse(f, events=("start", "end"))
        context = iter(context)
        # Get the root element
        event, root = next(context)

        for event, elem in context:
            if event == "end" and elem.tag == "EinheitWind":
                # Apply pre-filtering
                bundesland = elem.findtext("Bundesland")
                status = elem.findtext("EinheitBetriebsstatus") or elem.findtext("Einheitenbetriebsstatus")

                if bundesland in ("1400", "14") and status == "35":
                    # Handle coordinates
                    lon = to_float(elem.findtext("Laengengrad"))
                    lat = to_float(elem.findtext("Breitengrad"))

                    if lon is None or lat is None or pd.isna(lon) or pd.isna(lat):
                        plz = elem.findtext("Postleitzahl")
                        imputed = False
                        if plz:
                            plz_str = plz.strip().zfill(5)
                            if plz_str in plz_df.index:
                                row = plz_df.loc[plz_str]
                                if isinstance(row, pd.DataFrame):
                                    row = row.iloc[0]
                                lat = float(row["lat"])
                                lon = float(row["lng"])
                                imputed = True
                        if not imputed:
                            lat = fallback_lat
                            lon = fallback_lon

                    record = {
                        "unit_mastr_number": elem.findtext("EinheitMastrNummer"),
                        "last_update_date": elem.findtext("DatumLetzteAktualisierung"),
                        "longitude": lon,
                        "latitude": lat,
                        "commissioning_date": elem.findtext("Inbetriebnahmedatum"),
                        "final_decommission_date": elem.findtext("DatumEndgueltigeStilllegung"),
                        "gross_power": to_float(elem.findtext("Bruttoleistung")),
                        "net_nominal_power": to_float(elem.findtext("Nettonennleistung")),
                        "manufacturer": elem.findtext("Hersteller"),
                        "technology": elem.findtext("Technologie"),
                        "type_designation": elem.findtext("Typenbezeichnung"),
                        "hub_height": to_float(elem.findtext("Nabenhoehe")),
                        "rotor_diameter": to_float(elem.findtext("Rotordurchmesser")),
                    }
                    records.append(record)

                # Clear processed element and root children to free memory
                elem.clear()
                root.clear()

    logger.info(f"Extracted {len(records)} active wind units in Brandenburg from XML.")
    return records


def parse_solar_units(
    xml_dir: str,
    plz_df: pd.DataFrame,
    fallback_lat: float,
    fallback_lon: float,
) -> List[Dict[str, Any]]:
    """Memory-safely parses all solar XML files in a directory.

    Args:
        xml_dir: Directory containing the solar XML files.
        plz_df: German postal code lookup DataFrame.
        fallback_lat: Fallback latitude (centroid).
        fallback_lon: Fallback longitude (centroid).

    Returns:
        A list of parsed solar installation records.
    """
    search_pattern = os.path.join(xml_dir, "EinheitenSolar_*.xml")
    xml_files = sorted(glob.glob(search_pattern))

    if not xml_files:
        logger.warning(f"No solar XML files found matching pattern: {search_pattern}")
        return []

    logger.info(f"Found {len(xml_files)} solar XML files to parse.")
    records = []

    # Mapping dictionaries for catalog IDs
    azimuth_map = {
        "695": 0.0,    # Nord
        "696": 45.0,   # Nord-Ost
        "697": 90.0,   # Ost
        "698": 135.0,  # Süd-Ost
        "699": 180.0,  # Süd
        "700": 225.0,  # Süd-West
        "701": 270.0,  # West
        "702": 315.0,  # Nord-West
    }

    tilt_map = {
        "806": 90.0,   # 90 Grad (vertikal)
        "807": 75.0,   # 61 - 89 Grad
        "808": 50.5,   # 41 - 60 Grad
        "809": 30.5,   # 21 - 40 Grad
        "810": 12.5,   # 5 - 20 Grad
    }

    for file_path in xml_files:
        filename = os.path.basename(file_path)
        logger.info(f"Parsing solar units from: {filename}")
        file_records = 0

        with ProgressFile(file_path, desc=f"Parsing {filename}") as f:
            context = ET.iterparse(f, events=("start", "end"))
            context = iter(context)
            # Get the root element
            event, root = next(context)

            for event, elem in context:
                if event == "end" and elem.tag == "EinheitSolar":
                    # Apply pre-filtering
                    bundesland = elem.findtext("Bundesland")
                    status = elem.findtext("EinheitBetriebsstatus") or elem.findtext("Einheitenbetriebsstatus")

                    if bundesland in ("1400", "14") and status == "35":
                        # Handle coordinates
                        lon = to_float(elem.findtext("Laengengrad"))
                        lat = to_float(elem.findtext("Breitengrad"))

                        if lon is None or lat is None or pd.isna(lon) or pd.isna(lat):
                            plz = elem.findtext("Postleitzahl")
                            imputed = False
                            if plz:
                                plz_str = plz.strip().zfill(5)
                                if plz_str in plz_df.index:
                                    row = plz_df.loc[plz_str]
                                    if isinstance(row, pd.DataFrame):
                                        row = row.iloc[0]
                                    lat = float(row["lat"])
                                    lon = float(row["lng"])
                                    imputed = True
                            if not imputed:
                                lat = fallback_lat
                                lon = fallback_lon

                        art = elem.findtext("ArtDerSolaranlage")
                        ausrichtung = elem.findtext("Hauptausrichtung")
                        neigung = elem.findtext("HauptausrichtungNeigungswinkel")
                        speicher = elem.findtext("SpeicherAmGleichenOrt")

                        record = {
                            "unit_mastr_number": elem.findtext("EinheitMastrNummer"),
                            "last_update_date": elem.findtext("DatumLetzteAktualisierung"),
                            "longitude": lon,
                            "latitude": lat,
                            "commissioning_date": elem.findtext("Inbetriebnahmedatum"),
                            "final_decommission_date": elem.findtext("DatumEndgueltigeStilllegung"),
                            "gross_power": to_float(elem.findtext("Bruttoleistung")),
                            "net_nominal_power": to_float(elem.findtext("Nettonennleistung")),
                            "is_ground_mounted": True if art == "852" else False,
                            "azimuth": azimuth_map.get(ausrichtung) if ausrichtung in azimuth_map else None,
                            "tilt": tilt_map.get(neigung) if neigung in tilt_map else None,
                            "has_battery_storage": True if speicher == "1" else False,
                        }
                        records.append(record)
                        file_records += 1

                    # Clear processed element and root children to free memory
                    elem.clear()
                    root.clear()

        logger.info(f"Extracted {file_records} active solar units from {filename}.")

    logger.info(f"Extracted a total of {len(records)} active solar units in Brandenburg from XMLs.")
    return records


def filter_spatial_boundary(
    records: List[Dict[str, Any]],
    geojson_path: str,
) -> gpd.GeoDataFrame:
    """Converts records to a GeoDataFrame and filters geographically using the boundary.

    Args:
        records: A list of unit dictionaries.
        geojson_path: Path to the Brandenburg boundary GeoJSON file.

    Returns:
        A GeoDataFrame containing only the units falling inside Brandenburg boundary.
    """
    if not records:
        logger.warning("No records to perform spatial filtering.")
        return gpd.GeoDataFrame()

    df = pd.DataFrame(records)
    # Explicit conversion of longitude and latitude to numeric floats
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df = df.dropna(subset=["longitude", "latitude"])

    # Create GeoPandas GeoDataFrame
    geometry = gpd.points_from_xy(df["longitude"], df["latitude"])
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # Load Brandenburg boundary
    brandenburg_gdf = gpd.read_file(geojson_path)
    brandenburg_poly = brandenburg_gdf.geometry.iloc[0]

    # Mask points that are inside the Brandenburg boundary
    within_mask = gdf.geometry.within(brandenburg_poly)
    gdf_filtered = gdf[within_mask].copy()

    logger.info(
        f"Geospatial check: Filtered units from {len(gdf)} to {len(gdf_filtered)} "
        f"using the irregular boundary of Brandenburg."
    )
    return gdf_filtered


def run_parsing_pipeline(config_path: str = "conf/config.yaml") -> None:
    """Orchestrates the entire streaming parser pipeline.

    Args:
        config_path: Path to the YAML configuration file.
    """
    config = load_config(config_path)

    # 1. Load postal codes lookup
    plz_path = "data/external/de_postal_codes.csv"
    logger.info(f"Loading postal codes lookup database from: {plz_path}")
    plz_df = pd.read_csv(plz_path, index_col=0, dtype={0: str})

    # 2. Get fallback centroid
    geojson_path = config["open_meteo"]["openstreetmap"]["geojson_local_path"]
    fallback_lat, fallback_lon = get_brandenburg_centroid(geojson_path)

    # 3. Parse and filter Wind Turbines
    wind_xml = config["marktstammdatenregister"]["wind"]["local_path"]
    wind_records = parse_wind_units(wind_xml, plz_df, fallback_lat, fallback_lon)
    wind_gdf = filter_spatial_boundary(wind_records, geojson_path)

    # Map directly to final columns and save
    wind_final_cols = config["processing"]["wind_turbines"]["final_columns"]
    wind_final_df = pd.DataFrame(wind_gdf[wind_final_cols])

    # 4. Parse and filter Solar Installations
    solar_dir = config["marktstammdatenregister"]["solar"]["local_path"]
    solar_records = parse_solar_units(solar_dir, plz_df, fallback_lat, fallback_lon)
    solar_gdf = filter_spatial_boundary(solar_records, geojson_path)

    # Map directly to final columns and save
    solar_final_cols = config["processing"]["solar_installations"]["final_columns"]
    solar_final_df = pd.DataFrame(solar_gdf[solar_final_cols])

    # 5. Output raw parsed files to data/raw/
    os.makedirs("data/raw", exist_ok=True)
    wind_output_path = "data/raw/mastr_wind_brandenburg_raw.csv"
    solar_output_path = "data/raw/mastr_solar_brandenburg_raw.csv"

    wind_final_df.to_csv(wind_output_path, index=False, encoding="utf-8")
    solar_final_df.to_csv(solar_output_path, index=False, encoding="utf-8")

    logger.info(f"Successfully saved wind raw dataset ({len(wind_final_df)} units) to: {wind_output_path}")
    logger.info(f"Successfully saved solar raw dataset ({len(solar_final_df)} units) to: {solar_output_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    run_parsing_pipeline()
