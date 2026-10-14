"""
Crop Damage Risk Prediction Model.
Implements a Scikit-Learn ensemble model that predicts damage risk probability,
evaluates risk severity levels, classifies primary hazards, and outputs actionable alerts.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    SENSOR_FEATURES,
    ENGINEERED_FEATURES,
    ALL_FEATURES,
    RISK_LEVEL_NAMES,
    DEFAULT_ALERT_THRESHOLDS,
    ACTIONABLE_ADVISORIES
)
from feature_pipeline import engineer_features


class CropRiskModel:
    """
    ML Pipeline for Agricultural IoT Sensor & Weather Telemetry Risk Scoring.
    """

    def __init__(self, n_estimators: int = 150, max_depth: int = 10, random_state: int = 42):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth

        # Primary Risk Level Classifier (0: Low, 1: Moderate, 2: High)
        self.risk_classifier = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight="balanced",
            random_state=self.random_state,
        )

        # Hazard Diagnostician Classifier (identifies root cause: Drought, Frost, Disease, etc.)
        self.hazard_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            class_weight="balanced",
            random_state=self.random_state,
        )

        self.thresholds = DEFAULT_ALERT_THRESHOLDS.copy()
        self.is_fitted = False
        self.feature_names = ALL_FEATURES

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensures all raw and engineered features are present."""
        df_feat = engineer_features(df)
        return df_feat[self.feature_names]

    def fit(
        self,
        X_df: pd.DataFrame,
        y_risk_level: np.ndarray,
        y_hazard: Optional[np.ndarray] = None
    ) -> "CropRiskModel":
        """
        Fits the risk level model and optional hazard diagnostic model.
        """
        X_proc = self._prepare_data(X_df)

        # Fit primary risk severity model
        self.risk_classifier.fit(X_proc, y_risk_level)

        # Fit hazard type model if provided
        if y_hazard is not None:
            self.hazard_classifier.fit(X_proc, y_hazard)

        self.is_fitted = True
        return self

    def predict_probabilities(self, X_df: pd.DataFrame) -> np.ndarray:
        """
        Returns class probabilities for [Low, Moderate, High].
        """
        X_proc = self._prepare_data(X_df)
        return self.risk_classifier.predict_proba(X_proc)

    def compute_continuous_risk_score(self, probs: np.ndarray) -> np.ndarray:
        """
        Computes a continuous normalized risk score in [0.0, 1.0].
        Score = (P(Moderate) * 0.5) + (P(High) * 1.0)
        """
        # probs has columns [P(Low), P(Mod), P(High)]
        if probs.shape[1] == 3:
            return probs[:, 1] * 0.5 + probs[:, 2] * 1.0
        elif probs.shape[1] == 2:
            return probs[:, 1]
        return np.zeros(len(probs))

    def set_thresholds(self, moderate_threshold: float, critical_threshold: float):
        """
        Configures the operational alert thresholds.
        """
        assert 0.0 < moderate_threshold < critical_threshold <= 1.0, (
            "Thresholds must satisfy 0 < moderate < critical <= 1"
        )
        self.thresholds["moderate_threshold"] = float(moderate_threshold)
        self.thresholds["critical_threshold"] = float(critical_threshold)

    def evaluate_telemetry_payload(self, telemetry: Dict[str, float]) -> Dict[str, Any]:
        """
        Evaluates a single real-time sensor/weather reading:
        Applies feature engineering, ML inference, safety threshold checks,
        and returns an actionable advisory report.
        """
        single_df = pd.DataFrame([telemetry])
        X_proc = self._prepare_data(single_df)

        # Probabilities
        probs = self.risk_classifier.predict_proba(X_proc)[0]
        prob_low, prob_mod, prob_high = probs[0], probs[1], probs[2]

        # Continuous risk index
        risk_score = round(float(prob_mod * 0.5 + prob_high * 1.0), 3)

        # Primary hazard inference
        predicted_hazard = self.hazard_classifier.predict(X_proc)[0]

        # Safety override guardrails (domain expert rules for extreme telemetry)
        override_triggered = False
        override_reason = ""

        if telemetry.get("ambient_temp_c", 20.0) <= 0.0:
            override_triggered = True
            override_reason = "Hard Guardrail: Ambient sub-zero freezing temperature detected."
            risk_score = max(risk_score, 0.85)
            predicted_hazard = "FROST_CHILL_DAMAGE"
        elif telemetry.get("soil_moisture_pct", 50.0) < 12.0 and telemetry.get("ambient_temp_c", 20.0) > 36.0:
            override_triggered = True
            override_reason = "Hard Guardrail: Extreme soil desiccation combined with heat wave."
            risk_score = max(risk_score, 0.82)
            predicted_hazard = "DROUGHT_HEAT_STRESS"
        elif telemetry.get("soil_moisture_pct", 50.0) > 88.0 and telemetry.get("rainfall_forecast_mm", 0.0) > 60.0:
            override_triggered = True
            override_reason = "Hard Guardrail: Soil saturation coupled with torrential forecast."
            risk_score = max(risk_score, 0.80)
            predicted_hazard = "WATERLOGGING_EXCESS_RAIN"

        # Apply calibrated operational thresholds
        mod_th = self.thresholds["moderate_threshold"]
        crit_th = self.thresholds["critical_threshold"]

        if risk_score >= crit_th:
            alert_level = "HIGH"
            alert_badge = "[CRITICAL - RED]"
        elif risk_score >= mod_th:
            alert_level = "MODERATE"
            alert_badge = "[WARNING - YELLOW]"
        else:
            alert_level = "LOW"
            alert_badge = "[NORMAL - GREEN]"

        # Actionable advisory recommendation
        advisory_map = ACTIONABLE_ADVISORIES.get(predicted_hazard, ACTIONABLE_ADVISORIES["NORMAL"])
        advisory_text = advisory_map.get(alert_level, "Monitor crop sensor telemetry regularly.")

        # Identify top contributing features for explainability
        feature_importances = dict(zip(self.feature_names, self.risk_classifier.feature_importances_))
        sample_values = X_proc.iloc[0].to_dict()

        return {
            "risk_score": risk_score,
            "alert_level": alert_level,
            "alert_badge": alert_badge,
            "primary_hazard": predicted_hazard,
            "advisory": advisory_text,
            "thresholds_used": self.thresholds,
            "probabilities": {
                "low": round(float(prob_low), 4),
                "moderate": round(float(prob_mod), 4),
                "high": round(float(prob_high), 4),
            },
            "guardrail_override": {
                "triggered": override_triggered,
                "reason": override_reason,
            },
            "telemetry_inputs": telemetry,
            "engineered_indicators": {
                "vpd_kpa": round(float(sample_values["vpd_kpa"]), 3),
                "thermal_stress": round(float(sample_values["thermal_stress_index"]), 3),
                "drought_stress": round(float(sample_values["drought_stress_index"]), 3),
                "waterlogging_risk": round(float(sample_values["waterlogging_risk_index"]), 3),
                "disease_favorability": round(float(sample_values["disease_favorability"]), 3),
                "wind_lodging": round(float(sample_values["wind_lodging_index"]), 3),
            }
        }

    def get_feature_importances(self) -> pd.DataFrame:
        """Returns sorted feature importances from the primary risk classifier."""
        importances = self.risk_classifier.feature_importances_
        df_imp = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)
        return df_imp

    def save(self, filepath: str):
        """Saves model bundle to file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "CropRiskModel":
        """Loads model bundle from file."""
        return joblib.load(filepath)
