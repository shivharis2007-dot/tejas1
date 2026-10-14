# Crop Damage Risk Prediction Model & Alert Threshold System

An end-to-end Machine Learning pipeline that ingests IoT sensor telemetry and weather forecasts to predict multi-hazard crop damage risk and dynamically trigger calibrated agronomic alerts.

---

## 1. System Overview

Agricultural crops face compounding environmental stressors ranging from flash droughts and heatwaves to root-suffocating floods, unseasonal frosts, and humidity-driven fungal blights. This system combines **agronomic biophysical feature engineering** with **machine learning classification** and **calibrated decision thresholds** to deliver timely, actionable early warnings.

### Architecture Highlights:
- **Telemetry Ingestion**: Reads 8 core IoT sensor and weather station features.
- **Biophysical Feature Engineering**: Computes Vapor Pressure Deficit (VPD), thermal stress deviation, root-zone drought stress index, waterlogging risk index, disease favorability factor, and wind lodging vulnerability.
- **Dual Ensemble ML**:
  - **Risk Severity Classifier**: Random Forest estimating probability distributions over `[LOW, MODERATE, HIGH]`.
  - **Hazard Diagnostician**: Identifies root cause (`DROUGHT_HEAT_STRESS`, `WATERLOGGING_EXCESS_RAIN`, `FROST_CHILL_DAMAGE`, `FUNGAL_DISEASE_PRESSURE`, `WIND_LODGING_DAMAGE`).
- **Dynamic Threshold Calibration**: Optimizes decision boundaries ($\theta_{moderate}$, $\theta_{critical}$) to maximize safety recall ($\ge 95\%$) on severe hazards while curbing false-alarm fatigue.
- **Hybrid Safety Guardrails**: Hard threshold overrides for extreme environmental conditions (e.g., sub-zero frost, acute soil desiccation).

---

## 2. Sensor & Weather Telemetry Schema

| Telemetry Variable | Units | Description |
| :--- | :--- | :--- |
| `soil_moisture_pct` | % | Volumetric soil water content in crop root zone (0-100%) |
| `soil_temp_c` | °C | Soil temperature at 10cm depth (-5 to 45°C) |
| `leaf_wetness_hours` | Hours | Duration of free water present on crop canopy (0-24 hrs/day) |
| `ambient_temp_c` | °C | Ambient air temperature (-10 to 50°C) |
| `relative_humidity_pct`| % | Relative atmospheric humidity (10-100%) |
| `rainfall_forecast_mm` | mm | Expected precipitation accumulation over next 24-48 hrs |
| `wind_speed_kmh` | km/h | Peak sustained wind velocity (0-100 km/h) |
| `solar_radiation_mj` | MJ/m² | Global horizontal solar irradiance (0-35 MJ/m²/day) |

---

## 3. Agronomic Engineered Indicators

1. **Vapor Pressure Deficit (VPD in kPa)**:
   $$e_s = 0.61078 \times \exp\left(\frac{17.27 \times T}{T + 237.3}\right)$$
   $$e_a = e_s \times \frac{RH}{100}$$
   $$VPD = e_s - e_a$$
   - *Agronomic impact*: Measures drying demand of air. High VPD (> 2.5 kPa) causes stomatal closure and wilting; low VPD (< 0.3 kPa) in warm air prevents transpiration and promotes pathogen spore germination.
2. **Thermal Stress Index**: Measures degree deviations outside the physiological optimum (12°C - 33°C).
3. **Drought Stress Index**: Quantifies root water deficit scaled by evaporative atmospheric demand.
4. **Waterlogging / Anoxia Risk Index**: Interaction of saturated soils (>65%) with incoming heavy rainfall.
5. **Disease Favorability Index**: Canopy wetness hours combined with humidity (>60%) and temperature suitability for fungal spores (bell curve centered at 24°C).
6. **Wind Lodging Index**: Dynamic pressure of wind gusts scaled by soil softening from excessive moisture.

---

## 4. Alert Threshold Calibration

The model computes a continuous **Crop Damage Risk Index** $R \in [0.0, 1.0]$:
$$R = 0.5 \times P(\text{Moderate}) + 1.0 \times P(\text{High})$$

### Operational Tiers:

| Alert Tier | Badge | Default Threshold | Action Required |
| :--- | :--- | :--- | :--- |
| **LOW** | `[NORMAL - GREEN]` | $R < 0.21$ | Standard farm operations, routine monitoring |
| **MODERATE** | `[WARNING - YELLOW]` | $0.21 \le R < 0.71$ | Preemptive advisory (deficit irrigation, check drainage, fungicide preparation) |
| **HIGH** | `[CRITICAL - RED]` | $R \ge 0.71$ | Immediate emergency intervention (frost heaters, drainage pumps, emergency canopy cooling) |

### Optimization Criteria:
- **High-Risk Safety Target**: High Risk Recall $\ge 95\%$ on the test set to guarantee that destructive frost, flood, or heat shocks are never missed.
- **Missed Critical Events**: **0 / 259** (0% critical events misclassified as Normal).

---

## 5. Project File Structure

```
crop_risk_model/
├── config.py                 # Telemetry schemas, thresholds, hazard definitions & advisories
├── feature_pipeline.py       # Agronomic feature calculations (VPD, stress indices)
├── synthetic_data.py         # Realistic field telemetry simulator
├── model.py                  # CropRiskModel pipeline with ML and guardrails
├── threshold_optimizer.py    # Threshold calibration and safety evaluation logic
├── train_and_evaluate.py     # Training, cross-validation, and artifact serialization
├── predict_and_alert.py      # Real-time inference CLI and API helper
├── demo.py                   # 6-scenario agricultural field simulator
├── requirements.txt          # Python dependencies
└── crop_risk_model.joblib    # Serialized trained model artifact
```

---

## 6. How to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train Model & Calibrate Thresholds
```bash
python train_and_evaluate.py
```

### Step 3: Run Interactive Field Scenarios Demo
```bash
python demo.py
```

### Step 4: Run Single-Sample Telemetry Prediction via CLI
```bash
# Heatwave and soil desiccation test
python predict_and_alert.py --soil_moisture 15.0 --ambient_temp 38.0 --humidity 20.0

# Freezing frost condition test
python predict_and_alert.py --ambient_temp -1.5 --soil_moisture 40.0 --soil_temp 2.0
```
