"""
Model Training and Evaluation Pipeline.
Trains the crop damage risk model, benchmarks classification metrics,
calibrates alert thresholds, and serializes the trained model bundle.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from synthetic_data import generate_agricultural_dataset
from model import CropRiskModel
from threshold_optimizer import calibrate_thresholds, evaluate_alert_policy
from config import RISK_LEVEL_NAMES


def run_training_pipeline(
    n_samples: int = 6000,
    model_save_path: str = "crop_risk_model.joblib",
    target_high_recall: float = 0.95
) -> CropRiskModel:
    """
    Executes full data generation, model fitting, metric evaluation,
    threshold calibration, and artifact serialization.
    """
    print(f"Generating synthetic agricultural dataset ({n_samples} samples)...")
    dataset = generate_agricultural_dataset(n_samples=n_samples, random_state=42)

    # Feature inputs and targets
    X = dataset.drop(columns=["risk_score", "risk_level", "primary_hazard"])
    y_level = dataset["risk_level"].values
    y_hazard = dataset["primary_hazard"].values

    # Train / Test split (80% train, 20% test)
    X_train, X_test, y_train, y_test, y_haz_train, y_haz_test, df_train, df_test = train_test_split(
        X, y_level, y_hazard, dataset, test_size=0.20, random_state=42, stratify=y_level
    )

    print(f"Dataset split: Train={len(X_train)} samples, Test={len(X_test)} samples.")

    # Instantiate and fit model
    print("Training CropRiskModel (Random Forest ensemble)...")
    model = CropRiskModel(n_estimators=150, max_depth=10, random_state=42)
    model.fit(X_train, y_train, y_hazard=y_haz_train)

    # Model inference on test split
    probs_test = model.predict_probabilities(X_test)
    preds_test = np.argmax(probs_test, axis=1)
    risk_scores_test = model.compute_continuous_risk_score(probs_test)

    # Classification metrics
    target_names = [RISK_LEVEL_NAMES[i] for i in range(3)]
    report = classification_report(y_test, preds_test, target_names=target_names)
    cm = confusion_matrix(y_test, preds_test)

    print("\n" + "=" * 60)
    print("CLASSIFICATION PERFORMANCE REPORT (TEST SET):")
    print("=" * 60)
    print(report)

    print("CONFUSION MATRIX [Row: True, Col: Pred] (Low, Mod, High):")
    print(cm)

    # Calibrate alert thresholds
    print("\n" + "=" * 60)
    print("CALIBRATING OPERATIONAL ALERT THRESHOLDS...")
    print("=" * 60)
    calibration_result = calibrate_thresholds(
        y_true_levels=y_test,
        predicted_risk_scores=risk_scores_test,
        target_high_risk_recall=target_high_recall
    )

    recommended_th = calibration_result["recommended_thresholds"]
    mod_th = recommended_th["moderate_threshold"]
    crit_th = recommended_th["critical_threshold"]

    print(f"Optimal Moderate (Yellow Advisory) Threshold : {mod_th:.2f}")
    print(f"Optimal Critical (Red Alert) Threshold        : {crit_th:.2f}")

    policy_perf = calibration_result["policy_performance"]
    print("\nOperational Alert Policy Performance under Calibrated Thresholds:")
    print(f" - Overall 3-Tier Accuracy        : {policy_perf['overall_3tier_accuracy'] * 100:.2f}%")
    print(f" - Critical Alert Recall (Safety) : {policy_perf['critical_alert']['recall'] * 100:.2f}%")
    print(f" - Critical Alert Precision       : {policy_perf['critical_alert']['precision'] * 100:.2f}%")
    print(f" - Missed Critical Events         : {policy_perf['critical_alert']['missed_as_normal_count']} / {policy_perf['critical_alert']['total_critical_events']}")
    print(f" - Advisory Alert Recall          : {policy_perf['advisory_alert']['recall'] * 100:.2f}%")

    # Set calibrated thresholds on model
    model.set_thresholds(moderate_threshold=mod_th, critical_threshold=crit_th)

    # Feature Importance analysis
    print("\n" + "=" * 60)
    print("TOP AGRONOMIC RISK PREDICTORS (FEATURE IMPORTANCE):")
    print("=" * 60)
    imp_df = model.get_feature_importances()
    for _, row in imp_df.head(8).iterrows():
        print(f" - {row['feature']:<25}: {row['importance']:.4f}")

    # Serialize trained model artifact
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_save_path = os.path.join(script_dir, model_save_path)
    model.save(full_save_path)
    print(f"\nModel artifact successfully saved to: {full_save_path}")

    return model


if __name__ == "__main__":
    run_training_pipeline()
