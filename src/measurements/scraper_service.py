import os
import csv
import requests
from datetime import datetime, timezone
from loguru import logger
from omegaconf import OmegaConf

class ScraperService:
    def __init__(self, cfg: OmegaConf):
        self.cfg = cfg
        self.api_base_url = cfg.scraper.api_base_url
        self.tenant_id = cfg.scraper.tenant_id
        
        # New CSV header structure
        self.csv_headers = [
            "timestamp",
            "biomasse",
            "photovoltaik",
            "windkraft",
            "weitere_erzeuger",
            "regional_erzeugter_strom",
            "industrie_und_gewerbe",
            "kummunale_anlagen",
            "private_haushalte",
            "eigenversorgung",
            "prozent_regional_stromerzeugung_pv",
            "heute_eingespart_tCO2",
            "prozent_wolkenbedeckung"
        ]

    def _ensure_csv_header(self, csv_path: str):
        """
        Ensures that the CSV file exists and has the correct header.
        If it has the old header without a timestamp, it will overwrite it with the new header.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        write_header = False
        
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            write_header = True
        else:
            # Read first line to check if it matches the current expected header
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                
                # Check if first line contains 'timestamp'
                if "timestamp" not in first_line:
                    logger.warning(f"CSV file {csv_path} contains old header. Overwriting with new timestamped header.")
                    write_header = True
            except Exception as e:
                logger.error(f"Error reading CSV header for {csv_path}: {e}")
                write_header = True
                
        if write_header:
            try:
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                logger.info(f"Initialized CSV file {csv_path} with new header.")
            except Exception as e:
                logger.error(f"Failed to initialize CSV file {csv_path}: {e}")
                raise

    def scrape_region(self, region_name: str):
        """
        Scrapes meter and weather data for a specific region,
        processes the numbers, and appends a new record to its CSV file.
        """
        region_cfg = self.cfg.scraper.regions.get(region_name)
        if not region_cfg:
            logger.error(f"Region '{region_name}' is not defined in config.")
            return False
            
        code = region_cfg.code
        csv_path = region_cfg.csv_path
        
        logger.info(f"Starting scrape for region '{region_name}' (Code: {code})...")
        
        # 1. Fetch meter-data
        meter_url = f"{self.api_base_url}/meter-data?regionCode={code}"
        try:
            r = requests.get(meter_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            r.raise_for_status()
            meter_data = r.json()
        except Exception as e:
            logger.error(f"Failed to fetch meter-data for {region_name}: {e}")
            return False
            
        # 2. Fetch weather-data
        weather_url = f"{self.api_base_url}/weather-data?regionCode={code}&tenantId={self.tenant_id}"
        try:
            r = requests.get(weather_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            r.raise_for_status()
            weather_data = r.json()
        except Exception as e:
            logger.error(f"Failed to fetch weather-data for {region_name}: {e}")
            # We can still proceed if weather is down, but set default cloud cover
            weather_data = {}
            
        # 3. Parse fields robustly
        try:
            # Parse timestamp (using interval end)
            end_ts = int(meter_data.get("timestamp", {}).get("end", 0))
            if end_ts > 0:
                dt = datetime.fromtimestamp(end_ts, tz=timezone.utc)
                timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                
            # Parse feed-in list
            feed_in_items = meter_data.get("feedIn", {}).get("list", [])
            feed_in_map = {item["name"]: float(item.get("usage", 0.0)) for item in feed_in_items}
            
            biomasse = feed_in_map.get("bio", 0.0)
            photovoltaik = feed_in_map.get("solar", 0.0)
            windkraft = feed_in_map.get("wind", 0.0)
            weitere_erzeuger = feed_in_map.get("others", 0.0)
            
            # Total regional generation
            regional_erzeugter_strom = float(meter_data.get("feedIn", {}).get("total", 0.0))
            if regional_erzeugter_strom == 0.0:
                # Fallback to sum if total is missing
                regional_erzeugter_strom = biomasse + photovoltaik + windkraft + weitere_erzeuger
                
            # Parse consumptions list
            consumption_items = meter_data.get("consumptions", {}).get("list", [])
            consumption_map = {item["name"]: float(item.get("usage", 0.0)) for item in consumption_items}
            
            industrie_und_gewerbe = consumption_map.get("industrial", 0.0)
            kummunale_anlagen = consumption_map.get("public", 0.0)
            private_haushalte = consumption_map.get("domestic", 0.0)
            
            # Autarky (Eigenversorgung)
            eigenversorgung = float(meter_data.get("autarky", 0.0))
            
            # Percentage of PV in regional generation
            if regional_erzeugter_strom > 0:
                prozent_regional_stromerzeugung_pv = (photovoltaik / regional_erzeugter_strom) * 100.0
            else:
                prozent_regional_stromerzeugung_pv = 0.0
                
            # CO2 savings in tonnes (API returns in kg, e.g., 78352.7 -> 78.35 tonnes)
            heute_eingespart_tCO2 = float(meter_data.get("dailyCo2Savings", 0.0)) / 1000.0
            
            # Cloud cover in percentage (API returns fraction, e.g., 0.439 -> 43.9%)
            cloud_cover_fraction = float(weather_data.get("weather", {}).get("cloudCover", 0.0))
            prozent_wolkenbedeckung = cloud_cover_fraction * 100.0
            
            # Prepare row matching headers exactly
            row = [
                timestamp_str,
                biomasse,
                photovoltaik,
                windkraft,
                weitere_erzeuger,
                regional_erzeugter_strom,
                industrie_und_gewerbe,
                kummunale_anlagen,
                private_haushalte,
                eigenversorgung,
                prozent_regional_stromerzeugung_pv,
                heute_eingespart_tCO2,
                prozent_wolkenbedeckung
            ]
            
            # 4. Append to CSV
            self._ensure_csv_header(csv_path)
            
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row)
                
            logger.info(f"Successfully appended record for '{region_name}' to {csv_path} (Timestamp: {timestamp_str})")
            return True
            
        except Exception as e:
            logger.exception(f"Error parsing/saving data for region '{region_name}': {e}")
            return False
