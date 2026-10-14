"""
Feature Engineering Pipeline for Crop Damage Risk Modeling.
Transforms raw IoT sensor telemetry and meteorological forecasts into
agronomic stress indices (Vapor Pressure Deficit, Waterlogging Index, etc.).
"""

import math
import pandas as pd
import numpy as np
from typing import Dict, Union

def calculate_vpd(ambient_temp_c: float, relative_humidity_pct: float) -> float:
    """
    Calculate Vapor Pressure Deficit (VPD in kPa).
    VPD is the drying power of the air. High VPD indicates severe plant transpiration stress.
    Low VPD (< 0.3 kPa) in warm air triggers foliar fungal outbreaks.
    """
    # Tetens formula for saturation vapor pressure (kPa)
    es = 0.61078 * np.exp((17.27 * ambient_temp_c) / (ambient_temp_c + 237.3))
    ea = es * (relative_humidity_pct / 100.0)
    vpd = np.maximum(0.0, es - ea)
    return float(vpd)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes domain-specific agronomic features from raw sensor and weather telemetry.
    """
    data = df.copy()

    # 1. Vapor Pressure Deficit (VPD) in kPa
    # es = 0.61078 * exp(17.27 * T / (T + 237.3))
    t = data["ambient_temp_c"].values
    rh = np.clip(data["relative_humidity_pct"].values, 1.0, 100.0)
    es = 0.61078 * np.exp((17.27 * t) / (t + 237.3))
    ea = es * (rh / 100.0)
    data["vpd_kpa"] = np.maximum(0.0, es - ea)

    # 2. Thermal Stress Index:
    # Crops experience severe stress below 10°C (frost/chill) or above 34°C (heat shock)
    cold_stress = np.maximum(0.0, 12.0 - data["ambient_temp_c"].values)
    heat_stress = np.maximum(0.0, data["ambient_temp_c"].values - 33.0)
    data["thermal_stress_index"] = (cold_stress * 1.5) + (heat_stress * 1.2)

    # 3. Drought Stress Index:
    # Soil moisture deficit scaled by high evaporative demand (VPD and solar radiation)
    soil_dryness = np.maximum(0.0, 30.0 - data["soil_moisture_pct"].values) / 30.0
    evap_demand = (data["vpd_kpa"].values / 2.5) * (1.0 + data["solar_radiation_mj"].values / 25.0)
    data["drought_stress_index"] = np.clip(soil_dryness * evap_demand, 0.0, 5.0)

    # 4. Waterlogging / Anoxia Risk Index:
    # Soil moisture saturation (>75%) coupled with high forecast precipitation
    soil_saturation = np.maximum(0.0, data["soil_moisture_pct"].values - 65.0) / 35.0
    rain_weight = data["rainfall_forecast_mm"].values / 50.0
    data["waterlogging_risk_index"] = np.clip(soil_saturation * (1.0 + rain_weight), 0.0, 5.0)

    # 5. Disease Favorability Index:
    # Fungal germination thrives when leaf wetness > 6 hrs, RH > 75%, and temp is between 18-30°C
    temp_suitability = np.exp(-0.5 * ((data["ambient_temp_c"].values - 24.0) / 6.0) ** 2)
    wetness_factor = data["leaf_wetness_hours"].values / 24.0
    rh_factor = np.maximum(0.0, data["relative_humidity_pct"].values - 60.0) / 40.0
    data["disease_favorability"] = np.clip(wetness_factor * rh_factor * temp_suitability * 5.0, 0.0, 5.0)

    # 6. Wind Lodging Index:
    # Crop stem snapping or uprooting risk: wind speed squared multiplied by loose wet soil factor
    soil_softness = 0.5 + (data["soil_moisture_pct"].values / 100.0) * 0.5
    data["wind_lodging_index"] = ((data["wind_speed_kmh"].values / 50.0) ** 2) * soil_softness

    return data


def extract_single_payload_features(payload: Dict[str, float]) -> pd.DataFrame:
    """
    Transforms a single JSON telemetry record into a model-ready feature row.
    """
    df = pd.DataFrame([payload])
    return engineer_features(df)
