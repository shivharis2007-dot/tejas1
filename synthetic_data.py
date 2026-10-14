"""
Synthetic Agronomic Sensor and Weather Telemetry Generator.
Simulates realistic microclimate patterns, sensor noise, and multi-hazard agricultural stress events
(Drought/Heat, Waterlogging, Frost/Freeze, Fungal Blight, Wind Lodging).
"""

import numpy as np
import pandas as pd
from typing import Tuple
from config import SENSOR_FEATURES

def generate_agricultural_dataset(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic dataset simulating IoT farm sensors and meteorological forecasts.
    Returns a DataFrame containing raw sensor inputs, true continuous risk scores (0.0-1.0),
    categorical risk levels (0: Low, 1: Moderate, 2: High), and primary hazard labels.
    """
    np.random.seed(random_state)

    # 1. Base weather and soil conditions under standard temperate/subtropical growing seasons
    # Base distributions
    ambient_temp = np.random.normal(loc=23.0, scale=8.0, size=n_samples)
    soil_temp = ambient_temp * 0.85 + np.random.normal(loc=2.0, scale=2.5, size=n_samples)
    relative_humidity = np.clip(np.random.normal(loc=65.0, scale=18.0, size=n_samples), 15.0, 99.0)
    soil_moisture = np.clip(np.random.normal(loc=48.0, scale=18.0, size=n_samples), 5.0, 98.0)
    leaf_wetness = np.clip(np.random.exponential(scale=3.5, size=n_samples), 0.0, 24.0)
    rainfall_forecast = np.clip(np.random.exponential(scale=6.0, size=n_samples) - 2.0, 0.0, 150.0)
    wind_speed = np.clip(np.random.weibull(a=1.8, size=n_samples) * 16.0, 0.0, 95.0)
    solar_radiation = np.clip(np.random.normal(loc=18.0, scale=7.0, size=n_samples), 1.0, 32.0)

    # 2. Inject specific agricultural hazard archetypes to represent realistic field extremes
    hazards = ["NORMAL"] * n_samples
    primary_hazard_arr = np.array(hazards, dtype=object)

    # A. Frost / Chill Snap (Cold hazard) ~ 10% of samples
    frost_idx = np.random.choice(n_samples, size=int(n_samples * 0.10), replace=False)
    ambient_temp[frost_idx] = np.random.uniform(-4.0, 3.5, size=len(frost_idx))
    soil_temp[frost_idx] = np.random.uniform(0.0, 6.0, size=len(frost_idx))
    solar_radiation[frost_idx] = np.random.uniform(2.0, 12.0, size=len(frost_idx))
    primary_hazard_arr[frost_idx] = "FROST_CHILL_DAMAGE"

    # B. Drought / Scorching Heat Wave ~ 12% of samples
    remaining = [i for i in range(n_samples) if i not in frost_idx]
    drought_idx = np.random.choice(remaining, size=int(n_samples * 0.12), replace=False)
    ambient_temp[drought_idx] = np.random.uniform(34.0, 46.0, size=len(drought_idx))
    soil_temp[drought_idx] = ambient_temp[drought_idx] + np.random.uniform(2.0, 6.0, size=len(drought_idx))
    soil_moisture[drought_idx] = np.random.uniform(5.0, 18.0, size=len(drought_idx))
    relative_humidity[drought_idx] = np.random.uniform(15.0, 35.0, size=len(drought_idx))
    rainfall_forecast[drought_idx] = 0.0
    solar_radiation[drought_idx] = np.random.uniform(24.0, 32.0, size=len(drought_idx))
    primary_hazard_arr[drought_idx] = "DROUGHT_HEAT_STRESS"

    # C. Flood / Waterlogging / Excessive Monsoon Rain ~ 10% of samples
    remaining = [i for i in remaining if i not in drought_idx]
    flood_idx = np.random.choice(remaining, size=int(n_samples * 0.10), replace=False)
    soil_moisture[flood_idx] = np.random.uniform(78.0, 98.0, size=len(flood_idx))
    rainfall_forecast[flood_idx] = np.random.uniform(45.0, 140.0, size=len(flood_idx))
    relative_humidity[flood_idx] = np.random.uniform(85.0, 99.0, size=len(flood_idx))
    primary_hazard_arr[flood_idx] = "WATERLOGGING_EXCESS_RAIN"

    # D. Fungal Disease Outbreak microclimate ~ 12% of samples
    remaining = [i for i in remaining if i not in flood_idx]
    fungal_idx = np.random.choice(remaining, size=int(n_samples * 0.12), replace=False)
    leaf_wetness[fungal_idx] = np.random.uniform(9.0, 22.0, size=len(fungal_idx))
    relative_humidity[fungal_idx] = np.random.uniform(80.0, 98.0, size=len(fungal_idx))
    ambient_temp[fungal_idx] = np.random.uniform(20.0, 29.0, size=len(fungal_idx))
    primary_hazard_arr[fungal_idx] = "FUNGAL_DISEASE_PRESSURE"

    # E. Gale / Severe Wind Lodging ~ 8% of samples
    remaining = [i for i in remaining if i not in fungal_idx]
    wind_idx = np.random.choice(remaining, size=int(n_samples * 0.08), replace=False)
    wind_speed[wind_idx] = np.random.uniform(50.0, 92.0, size=len(wind_idx))
    primary_hazard_arr[wind_idx] = "WIND_LODGING_DAMAGE"

    # Construct base dataframe
    df = pd.DataFrame({
        "soil_moisture_pct": np.round(soil_moisture, 2),
        "soil_temp_c": np.round(soil_temp, 2),
        "leaf_wetness_hours": np.round(leaf_wetness, 1),
        "ambient_temp_c": np.round(ambient_temp, 2),
        "relative_humidity_pct": np.round(relative_humidity, 1),
        "rainfall_forecast_mm": np.round(rainfall_forecast, 1),
        "wind_speed_kmh": np.round(wind_speed, 1),
        "solar_radiation_mj": np.round(solar_radiation, 2),
        "primary_hazard": primary_hazard_arr,
    })

    # 3. Calculate latent crop damage risk function based on agronomic biophysics
    # Sub-component damage risks (each scaled roughly 0 to 1)
    
    # Drought & thermal component
    drought_component = np.clip(
        (np.maximum(0, 25.0 - df["soil_moisture_pct"]) / 22.0) * 0.55 +
        (np.maximum(0, df["ambient_temp_c"] - 34.0) / 10.0) * 0.45,
        0.0, 1.0
    )

    # Frost component
    frost_component = np.clip(
        np.maximum(0, 4.0 - df["ambient_temp_c"]) / 6.0,
        0.0, 1.0
    )

    # Waterlogging component
    waterlog_component = np.clip(
        (np.maximum(0, df["soil_moisture_pct"] - 72.0) / 25.0) * 0.5 +
        (np.maximum(0, df["rainfall_forecast_mm"] - 30.0) / 70.0) * 0.5,
        0.0, 1.0
    )

    # Fungal pathogen component
    fungal_component = np.clip(
        (np.maximum(0, df["leaf_wetness_hours"] - 7.0) / 14.0) * 0.5 +
        (np.maximum(0, df["relative_humidity_pct"] - 75.0) / 23.0) * 0.35 +
        (1.0 - np.clip(np.abs(df["ambient_temp_c"] - 24.0) / 7.0, 0.0, 1.0)) * 0.15,
        0.0, 1.0
    )

    # Wind lodging component
    wind_component = np.clip(
        (np.maximum(0, df["wind_speed_kmh"] - 45.0) / 45.0) * 0.7 +
        (df["soil_moisture_pct"] / 100.0) * 0.3,
        0.0, 1.0
    )

    # Combine maximum hazard impact with interaction baseline and sensory noise
    raw_max_risk = np.maximum.reduce([
        drought_component,
        frost_component,
        waterlog_component,
        fungal_component,
        wind_component
    ])

    # Add realistic environmental stochastic noise
    noise = np.random.normal(0, 0.04, size=n_samples)
    continuous_risk = np.clip(raw_max_risk + noise, 0.0, 1.0)
    df["risk_score"] = np.round(continuous_risk, 3)

    # Determine ground truth risk level (0: Low, 1: Moderate, 2: High)
    risk_level = np.zeros(n_samples, dtype=int)
    risk_level[df["risk_score"] >= 0.35] = 1
    risk_level[df["risk_score"] >= 0.70] = 2
    df["risk_level"] = risk_level

    # Update primary hazard if risk is truly low
    df.loc[df["risk_score"] < 0.30, "primary_hazard"] = "NORMAL"

    return df


if __name__ == "__main__":
    df = generate_agricultural_dataset(20)
    print("Sample generated dataset:")
    print(df.head())
    print("\nRisk Level Distribution:")
    print(df["risk_level"].value_counts())
