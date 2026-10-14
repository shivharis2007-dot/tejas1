"""
Alert Threshold Optimization and Calibration Module.
Calibrates operational decision boundaries (Moderate & Critical thresholds)
to balance early warning sensitivity (high recall) against false-alarm fatigue.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any
from sklearn.metrics import precision_recall_curve, fbeta_score, precision_score, recall_score, f1_score


def calibrate_thresholds(
    y_true_levels: np.ndarray,
    predicted_risk_scores: np.ndarray,
    target_high_risk_recall: float = 0.95,
    beta: float = 2.0
) -> Dict[str, Any]:
    """
    Calibrates operational alert thresholds on validation data.
    
    Parameters:
    - y_true_levels: Ground truth ordinal risk levels (0: Low, 1: Moderate, 2: High)
    - predicted_risk_scores: Continuous model-predicted risk scores in [0.0, 1.0]
    - target_high_risk_recall: Desired recall rate for catastrophic high-risk damage (default 0.95)
    - beta: F-beta weight emphasizing recall over precision for agricultural protection (default 2.0)
    
    Returns:
    - Dict with recommended thresholds, performance metrics, and threshold sweep table.
    """
    # 1. Binary problem for Moderate+High (Any Adverse Risk vs Normal)
    y_any_adverse = (y_true_levels >= 1).astype(int)
    
    # 2. Binary problem for Critical High Risk (High vs Low/Moderate)
    y_critical = (y_true_levels == 2).astype(int)

    # Threshold sweep grid
    candidate_thresholds = np.linspace(0.20, 0.85, 66)

    # --- Optimize Critical Alert Threshold ---
    # In agriculture, failing to alert for a freeze or flash flood is catastrophic.
    # We find the highest threshold that meets or exceeds target_high_risk_recall.
    crit_metrics = []
    best_crit_th = 0.70
    min_prec_at_target_recall = 0.0

    for th in candidate_thresholds:
        preds = (predicted_risk_scores >= th).astype(int)
        rec = recall_score(y_critical, preds, zero_division=0)
        prec = precision_score(y_critical, preds, zero_division=0)
        f_b = fbeta_score(y_critical, preds, beta=beta, zero_division=0)
        f_1 = f1_score(y_critical, preds, zero_division=0)
        crit_metrics.append({
            "threshold": round(float(th), 2),
            "recall": round(float(rec), 3),
            "precision": round(float(prec), 3),
            "f2_score": round(float(f_b), 3),
            "f1_score": round(float(f_1), 3),
        })

    crit_df = pd.DataFrame(crit_metrics)
    
    # Select threshold that meets target recall with highest precision
    valid_crit = crit_df[crit_df["recall"] >= target_high_risk_recall]
    if not valid_crit.empty:
        best_crit_idx = valid_crit["precision"].idxmax()
        best_crit_th = valid_crit.loc[best_crit_idx, "threshold"]
    else:
        # Fallback to threshold that maximizes F-beta
        best_crit_idx = crit_df["f2_score"].idxmax()
        best_crit_th = crit_df.loc[best_crit_idx, "threshold"]

    # --- Optimize Moderate Advisory Threshold ---
    # Advisory alerts need good balance to warn farmers of developing stress (F1 / F2 max on any adverse)
    mod_metrics = []
    for th in candidate_thresholds[candidate_thresholds < best_crit_th]:
        preds = (predicted_risk_scores >= th).astype(int)
        rec = recall_score(y_any_adverse, preds, zero_division=0)
        prec = precision_score(y_any_adverse, preds, zero_division=0)
        f_b = fbeta_score(y_any_adverse, preds, beta=1.5, zero_division=0)
        f_1 = f1_score(y_any_adverse, preds, zero_division=0)
        mod_metrics.append({
            "threshold": round(float(th), 2),
            "recall": round(float(rec), 3),
            "precision": round(float(prec), 3),
            "f1.5_score": round(float(f_b), 3),
            "f1_score": round(float(f_1), 3),
        })

    mod_df = pd.DataFrame(mod_metrics)
    # Select threshold maximizing F1.5 on adverse detection
    if not mod_df.empty:
        best_mod_idx = mod_df["f1.5_score"].idxmax()
        best_mod_th = mod_df.loc[best_mod_idx, "threshold"]
    else:
        best_mod_th = 0.35

    # Ensure gap of at least 0.20 between moderate and critical
    if (best_crit_th - best_mod_th) < 0.20:
        best_mod_th = max(0.25, round(best_crit_th - 0.25, 2))

    # Evaluate final policy metrics
    final_policy_eval = evaluate_alert_policy(
        y_true_levels,
        predicted_risk_scores,
        moderate_th=best_mod_th,
        critical_th=best_crit_th
    )

    return {
        "recommended_thresholds": {
            "moderate_threshold": float(best_mod_th),
            "critical_threshold": float(best_crit_th),
        },
        "target_high_risk_recall": target_high_risk_recall,
        "policy_performance": final_policy_eval,
        "critical_sweep_sample": crit_df.iloc[::8].to_dict(orient="records"),
    }


def evaluate_alert_policy(
    y_true: np.ndarray,
    risk_scores: np.ndarray,
    moderate_th: float,
    critical_th: float
) -> Dict[str, Any]:
    """
    Evaluates 3-tier classification performance under specific threshold configuration:
    - Tier 0: Normal / Green (< moderate_th)
    - Tier 1: Warning / Yellow (moderate_th <= score < critical_th)
    - Tier 2: Critical / Red (>= critical_th)
    """
    preds = np.zeros(len(risk_scores), dtype=int)
    preds[risk_scores >= moderate_th] = 1
    preds[risk_scores >= critical_th] = 2

    accuracy = np.mean(preds == y_true)
    
    # Critical Alert Performance (Red)
    y_crit_true = (y_true == 2).astype(int)
    y_crit_pred = (preds == 2).astype(int)
    crit_rec = recall_score(y_crit_true, y_crit_pred, zero_division=0)
    crit_prec = precision_score(y_crit_true, y_crit_pred, zero_division=0)
    
    # Any Alert Performance (Yellow + Red)
    y_adv_true = (y_true >= 1).astype(int)
    y_adv_pred = (preds >= 1).astype(int)
    adv_rec = recall_score(y_adv_true, y_adv_pred, zero_division=0)
    adv_prec = precision_score(y_adv_true, y_adv_pred, zero_division=0)

    # Missed Critical Events (Safety Failure)
    missed_critical_count = int(np.sum((y_true == 2) & (preds == 0)))
    total_critical_count = int(np.sum(y_true == 2))

    return {
        "overall_3tier_accuracy": round(float(accuracy), 4),
        "critical_alert": {
            "threshold": critical_th,
            "recall": round(float(crit_rec), 4),
            "precision": round(float(crit_prec), 4),
            "missed_as_normal_count": missed_critical_count,
            "total_critical_events": total_critical_count,
        },
        "advisory_alert": {
            "threshold": moderate_th,
            "recall": round(float(adv_rec), 4),
            "precision": round(float(adv_prec), 4),
        }
    }
