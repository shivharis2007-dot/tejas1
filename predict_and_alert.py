"""
Inference and Real-Time Alert Dispatcher.
Loads trained model artifact, evaluates incoming IoT sensor & weather readings,
and produces alert notifications with agronomic recommendations.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any

from model import CropRiskModel


def load_model(model_path: str = None) -> CropRiskModel:
    """Loads the trained CropRiskModel artifact."""
    if model_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "crop_risk_model.joblib")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. "
            f"Please run 'python train_and_evaluate.py' first."
        )
    return CropRiskModel.load(model_path)


def predict_risk(telemetry_data: Dict[str, float], model: CropRiskModel = None) -> Dict[str, Any]:
    """
    Evaluates incoming telemetry data and returns risk score and alert details.
    """
    if model is None:
        model = load_model()
    return model.evaluate_telemetry_payload(telemetry_data)


def format_alert_report(result: Dict[str, Any]) -> str:
    """Formats the evaluation result into a clean terminal report."""
    badge = result["alert_badge"]
    level = result["alert_level"]
    score = result["risk_score"]
    hazard = result["primary_hazard"]
    advisory = result["advisory"]
    probs = result["probabilities"]
    eng = result["engineered_indicators"]

    lines = [
        "=" * 65,
        f" CROP DAMAGE RISK ASSESSMENT: {badge}",
        "=" * 65,
        f" Risk Index Score : {score:.3f} / 1.000",
        f" Severity Tier    : {level}",
        f" Primary Hazard   : {hazard}",
        "-" * 65,
        " Class Probabilities:",
        f"   - Low (Normal) : {probs['low'] * 100:.1f}%",
        f"   - Moderate     : {probs['moderate'] * 100:.1f}%",
        f"   - High Risk    : {probs['high'] * 100:.1f}%",
        "-" * 65,
        " Key Microclimate Indicators:",
        f"   - Vapor Pressure Deficit (VPD) : {eng['vpd_kpa']} kPa",
        f"   - Thermal Stress Index         : {eng['thermal_stress']}",
        f"   - Drought Stress Index         : {eng['drought_stress']}",
        f"   - Waterlogging Risk Index      : {eng['waterlogging_risk']}",
        f"   - Fungal Disease Favorability  : {eng['disease_favorability']}",
        f"   - Wind Lodging Index           : {eng['wind_lodging']}",
    ]

    if result["guardrail_override"]["triggered"]:
        lines.append("-" * 65)
        lines.append(f" [!] SAFETY GUARDRAIL TRIGGERED: {result['guardrail_override']['reason']}")

    lines.extend([
        "=" * 65,
        " ACTIONABLE AGRONOMIC ADVISORY:",
        f" >> {advisory}",
        "=" * 65,
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop Damage Risk Prediction Inference")
    parser.add_argument("--json", type=str, help="Raw JSON string containing telemetry features")
    parser.add_argument("--soil_moisture", type=float, default=22.0)
    parser.add_argument("--ambient_temp", type=float, default=38.5)
    parser.add_argument("--humidity", type=float, default=25.0)
    parser.add_argument("--leaf_wetness", type=float, default=1.0)
    parser.add_argument("--rainfall", type=float, default=0.0)
    parser.add_argument("--wind", type=float, default=18.0)
    parser.add_argument("--solar", type=float, default=28.0)
    parser.add_argument("--soil_temp", type=float, default=32.0)

    args = parser.parse_args()

    if args.json:
        telemetry = json.loads(args.json)
    else:
        telemetry = {
            "soil_moisture_pct": args.soil_moisture,
            "soil_temp_c": args.soil_temp,
            "leaf_wetness_hours": args.leaf_wetness,
            "ambient_temp_c": args.ambient_temp,
            "relative_humidity_pct": args.humidity,
            "rainfall_forecast_mm": args.rainfall,
            "wind_speed_kmh": args.wind,
            "solar_radiation_mj": args.solar,
        }

    model = load_model()
    eval_result = predict_risk(telemetry, model=model)
    print(format_alert_report(eval_result))
