"""Physical Wind Power Calculation and Height-Extrapolation Module.

This module implements physical wind energy models. It extrapolates wind speeds
from a reference height to turbine-specific hub heights using the logarithmic
wind profile (approximated via the Hellmann power law) and translates the 
extrapolated wind speeds into theoretical power output (kW) using matched 
manufacturer-specific power curves and 1D spline interpolation.
"""

import os
import re
import logging
import unittest
from pathlib import Path
from typing import Dict, Any, Union, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from src.data.boundary_fetcher import load_config

logger = logging.getLogger(__name__)


class WindPowerCalculator:
    """Calculates physical wind power outputs with height extrapolation and power curves.

    Attributes:
        config: Loaded project configuration dictionary.
        curves_df: Raw power curves DataFrame.
        speed_cols: Column names representing wind speeds in the curves DataFrame.
        wind_speeds: Array of numerical wind speeds (m/s) corresponding to columns.
        mfr_map: Mapping of MaStR manufacturer ID codes to names in the database.
    """

    def __init__(
        self,
        power_curves_path: str = "data/wind/power_curves.csv",
        config_path: str = "conf/config.yaml",
    ) -> None:
        """Initializes the WindPowerCalculator.

        Args:
            power_curves_path: Filepath to the power curves lookup sheet.
            config_path: Filepath to the YAML configuration file.

        Raises:
            FileNotFoundError: If the power curves file or config file is missing.
        """
        # Load config
        self.config = load_config(config_path)

        # Load power curves
        path = Path(power_curves_path)
        if not path.exists():
            logger.error(f"Power curves CSV not found at: {power_curves_path}")
            raise FileNotFoundError(f"Power curves CSV not found at: {power_curves_path}")

        # Note: power_curves.csv uses single quotes as field delimiters
        self.curves_df = pd.read_csv(path, quotechar="'")

        # Extract columns representing wind speed power values (e.g. "kW at 4.5 m/s")
        self.speed_cols = [c for c in self.curves_df.columns if "kW at" in c]
        
        # Parse numerical wind speeds from column names
        self.wind_speeds = np.array(
            [float(re.search(r"kW at ([\d\.]+) m/s", col).group(1)) for col in self.speed_cols]
        )

        # Precalculate maximum rated power for each turbine model in the database
        self.curves_df["rated_power_kw"] = self.curves_df[self.speed_cols].astype(float).max(axis=1)

        # Precompute normalized alphanumeric strings to speed up matching
        self.curves_df["clean_mfr"] = (
            self.curves_df["Manufacturer name"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.replace(r"[^a-z0-9]", "", regex=True)
        )
        self.curves_df["clean_name"] = (
            self.curves_df["Turbine name"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.replace(r"[^a-z0-9]", "", regex=True)
        )

        # Standard Marktstammdatenregister manufacturer code mapping dictionary
        self.mfr_map = {
            1660: "Vestas",
            1586: "Enercon",
            1627: "Nordex",
            1625: "Neg Micon",
            1597: "GE Energy",
            1645: "Senvion",
            2888: "Repower",
            1593: "Fuhrländer",
            1598: "GE Energy",
            1001682: "Nordex",
            1584: "Siemens",
            1628: "Nordex",
            1587: "Enercon",
            1596: "Gamesa",
            2889: "Repower",
            1657: "Senvion",
            1666: "Siemens",
        }

    def extrapolate_wind_speed(
        self, v_100m: Union[float, np.ndarray], hub_height: float
    ) -> Union[float, np.ndarray]:
        """Extrapolates wind speed from 100m to turbine hub height using the Hellmann Power Law.

        Args:
            v_100m: Wind speed(s) at 100m reference height (m/s).
            hub_height: Turbine hub height (m).

        Returns:
            Extrapolated wind speed(s) at hub height (m/s).
        """
        # Read exponent dynamically from configuration (default to 0.14 if missing)
        alpha = self.config.get("wind_power_calculation", {}).get("alpha", 0.14)
        
        # Power Law profile: v = v_ref * (z / z_ref) ** alpha
        return v_100m * (hub_height / 100.0) ** alpha

    def match_turbine_curve(
        self,
        manufacturer_id: Optional[float],
        type_designation: str,
        gross_power: Optional[float] = None,
        rotor_diameter: Optional[float] = None,
    ) -> pd.Series:
        """Matches turbine metadata from MaStR to a power curve row in the database.

        Uses a multi-tier scored heuristic based on manufacturer name, capacity (kW),
        rotor diameter (m), and name substring tokens.

        Args:
            manufacturer_id: The manufacturer ID code from MaStR.
            type_designation: The turbine type designation string.
            gross_power: The gross nominal power capacity in kW.
            rotor_diameter: The rotor diameter in meters.

        Returns:
            A pandas Series representing the matched row from the power curves database.

        Raises:
            ValueError: If no suitable match can be made and no capacity is provided for fallback.
        """
        # Get manufacturer name if mapped
        mfr_name = ""
        if manufacturer_id is not None and not pd.isna(manufacturer_id):
            mfr_name = self.mfr_map.get(int(manufacturer_id), "")

        # Clean type designation string
        clean_td = re.sub(r"[^a-z0-9]", "", str(type_designation).lower())

        best_score = -1
        best_row = None

        # Filter database to search only the correct manufacturer first
        mfr_filtered = pd.DataFrame()
        if mfr_name:
            clean_mfr_query = re.sub(r"[^a-z0-9]", "", mfr_name.lower())
            mfr_filtered = self.curves_df[
                self.curves_df["clean_mfr"].str.contains(clean_mfr_query)
            ]

        # Use filtered candidate pool, fallback to entire database if empty
        candidates = mfr_filtered if not mfr_filtered.empty else self.curves_df

        for idx, row in candidates.iterrows():
            score = 0
            c_mfr = row["clean_mfr"]
            c_name = row["clean_name"]
            c_rated = row["rated_power_kw"]

            # 1. Manufacturer match
            if mfr_name:
                clean_mfr_query = re.sub(r"[^a-z0-9]", "", mfr_name.lower())
                if clean_mfr_query in c_mfr or c_mfr in clean_mfr_query:
                    score += 10

            # 2. Capacity match (within 15%)
            if gross_power is not None and not pd.isna(gross_power) and gross_power > 0:
                if abs(c_rated - gross_power) < 0.15 * gross_power:
                    score += 5.0 * (1.0 - abs(c_rated - gross_power) / (0.15 * gross_power))

            # 3. Rotor diameter match (within 10%)
            c_rotor_match = re.search(r"\d+", row["Turbine name"])
            if c_rotor_match and rotor_diameter is not None and not pd.isna(rotor_diameter) and rotor_diameter > 0:
                c_rotor_val = float(c_rotor_match.group())
                if abs(c_rotor_val - rotor_diameter) < 0.10 * rotor_diameter:
                    score += 5.0 * (1.0 - abs(c_rotor_val - rotor_diameter) / (0.10 * rotor_diameter))

            # 4. Name similarity / token match
            if clean_td and c_name:
                if c_name in clean_td or clean_td in c_name:
                    score += 8

            if score > best_score:
                best_score = score
                best_row = row

        # If matching score is poor (< 15) and we restricted to manufacturer, try searching all curves
        if best_score < 15 and candidates is not self.curves_df:
            for idx, row in self.curves_df.iterrows():
                score = 0
                c_name = row["clean_name"]
                c_rated = row["rated_power_kw"]

                if clean_td and c_name:
                    if c_name in clean_td or clean_td in c_name:
                        score += 8

                if gross_power is not None and not pd.isna(gross_power) and gross_power > 0:
                    if abs(c_rated - gross_power) < 0.15 * gross_power:
                        score += 5.0 * (1.0 - abs(c_rated - gross_power) / (0.15 * gross_power))

                c_rotor_match = re.search(r"\d+", row["Turbine name"])
                if c_rotor_match and rotor_diameter is not None and not pd.isna(rotor_diameter) and rotor_diameter > 0:
                    c_rotor_val = float(c_rotor_match.group())
                    if abs(c_rotor_val - rotor_diameter) < 0.10 * rotor_diameter:
                        score += 5.0 * (1.0 - abs(c_rotor_val - rotor_diameter) / (0.10 * rotor_diameter))

                if score > best_score:
                    best_score = score
                    best_row = row

        # If score remains extremely poor, fall back to the closest capacity turbine in the database
        if best_row is None or best_score < 5:
            if gross_power is not None and not pd.isna(gross_power):
                closest_idx = (self.curves_df["rated_power_kw"] - gross_power).abs().idxmin()
                best_row = self.curves_df.loc[closest_idx]
                logger.warning(
                    f"Poor match score for turbine '{type_designation}' (mfr ID: {manufacturer_id}). "
                    f"Falling back to closest capacity match in database: "
                    f"'{best_row['Manufacturer name']} {best_row['Turbine name']}' "
                    f"({best_row['rated_power_kw']} kW vs registered {gross_power} kW)."
                )
            else:
                raise ValueError(
                    f"Unable to match turbine curve for type '{type_designation}' "
                    f"(mfr ID: {manufacturer_id}) and no fallback gross_power was provided."
                )

        return best_row

    def get_interpolator(self, curve_row: pd.Series, kind: str = "linear") -> interp1d:
        """Creates a 1D interpolator that maps wind speeds to turbine power output.

        Args:
            curve_row: A pandas Series representing the matched turbine power curve row.
            kind: Type of interpolation, e.g. 'linear' or 'cubic'. Defaults to 'linear'.

        Returns:
            A scipy.interpolate.interp1d instance.
        """
        power_values = curve_row[self.speed_cols].values.astype(float)
        
        # Bounding output to 0.0 outside the [0, 35] m/s wind speed range
        return interp1d(
            x=self.wind_speeds,
            y=power_values,
            kind=kind,
            bounds_error=False,
            fill_value=0.0,
        )

    def calculate_power(
        self,
        v_100m: Union[float, np.ndarray],
        hub_height: float,
        manufacturer_id: Optional[float],
        type_designation: str,
        gross_power: float,
        rotor_diameter: Optional[float] = None,
        interpolation_kind: str = "linear",
    ) -> Union[float, np.ndarray]:
        """Calculates theoretical electrical power output (kW) for a turbine.

        Args:
            v_100m: Wind speed(s) at 100m height (m/s).
            hub_height: Hub height of the turbine (m).
            manufacturer_id: Manufacturer ID from MaStR.
            type_designation: Type designation from MaStR.
            gross_power: Nominal capacity of the turbine (kW).
            rotor_diameter: Rotor diameter of the turbine (m).
            interpolation_kind: Scipy interpolation type ('linear' or 'cubic').

        Returns:
            Theoretical turbine electrical power generation in kW, bounded by 0 and gross_power.
        """
        # 1. Height extrapolation
        v_hub = self.extrapolate_wind_speed(v_100m, hub_height)

        # 2. Retrieve matched manufacturer power curve
        curve_row = self.match_turbine_curve(
            manufacturer_id=manufacturer_id,
            type_designation=type_designation,
            gross_power=gross_power,
            rotor_diameter=rotor_diameter,
        )

        # 3. Interpolate power output
        interpolator = self.get_interpolator(curve_row, kind=interpolation_kind)
        power_output = interpolator(v_hub)

        # 4. Strict bounding: capacity is capped between 0 and gross_power
        # Safeguard: if gross_power is missing, cap by the power curve rated capacity
        max_power = gross_power
        if max_power is None or pd.isna(max_power) or max_power <= 0:
            max_power = curve_row["rated_power_kw"]

        return np.clip(power_output, 0.0, max_power)


# =====================================================================
# Unit Tests
# =====================================================================

class TestWindPowerCalculator(unittest.TestCase):
    """Unit tests for validating the WindPowerCalculator physics and interpolator."""

    def setUp(self) -> None:
        """Sets up the calculator instance for testing."""
        self.calculator = WindPowerCalculator()

    def test_zero_wind_speed(self) -> None:
        """Verifies that 0 m/s wind speed yields exactly 0 kW power output."""
        power = self.calculator.calculate_power(
            v_100m=0.0,
            hub_height=100.0,  # Avoid extrapolation scaling for simplicity
            manufacturer_id=1660,  # Vestas
            type_designation="V90-2MW",
            gross_power=2000.0,
            rotor_diameter=90.0,
        )
        self.assertAlmostEqual(float(power), 0.0)

    def test_rated_wind_speed(self) -> None:
        """Verifies that rated wind speed (e.g. 13 m/s) yields max capacity."""
        power = self.calculator.calculate_power(
            v_100m=13.0,
            hub_height=100.0,
            manufacturer_id=1660,
            type_designation="V90-2MW",
            gross_power=2000.0,
            rotor_diameter=90.0,
        )
        # Yield should be close to nominal rated capacity (2000 kW)
        self.assertAlmostEqual(float(power), 2000.0, delta=10.0)

    def test_cut_out_wind_speed(self) -> None:
        """Verifies that exceeding the cut-out speed (e.g. 25 m/s) drops output to 0 kW."""
        # For Vestas V90/2000, power drops to zero above 25 m/s
        power = self.calculator.calculate_power(
            v_100m=26.0,
            hub_height=100.0,
            manufacturer_id=1660,
            type_designation="V90-2MW",
            gross_power=2000.0,
            rotor_diameter=90.0,
        )
        self.assertAlmostEqual(float(power), 0.0)

    def test_height_extrapolation(self) -> None:
        """Verifies height extrapolation logic increases wind speed with height."""
        # Test case: 10 m/s at 100m extrapolated to 150m
        v_hub = self.calculator.extrapolate_wind_speed(10.0, 150.0)
        # Expected v_hub for alpha=0.14: 10 * (150/100)^0.14 = 10.584 m/s
        self.assertGreater(v_hub, 10.0)
        self.assertAlmostEqual(v_hub, 10.584, places=3)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
