"""Multi-horizon evaluation engine for attack forecasting and state dynamics.

Metrics:
    - Attack Risk Forecasting (per horizon h in 1..4):
        * Precision, Recall, F1-Score (at standard 0.5 and optimal threshold)
        * False Positive Rate (FPR = FP / (FP + TN))
        * PR-AUC (Average Precision score)
        * ROC-AUC
        * Brier Score (mean squared probability error)
        * Class counts and undefined metric tracking
    - State Dynamics Rollout (World Model only):
        * Mean Squared Error (MSE)
        * Mean Absolute Error (MAE)
        * Feature Cosine Similarity
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.contracts import FORECAST_STEPS


def _binary_probabilities(probs: Any) -> np.ndarray:
    """Ensure array has float probabilities clipped to [0, 1]."""
    arr = np.asarray(probs, dtype=np.float64)
    return np.clip(arr, 0.0, 1.0)


def evaluate_model_predictions(
    y_true: Any,
    y_probs: Any,
    model_name: str = "Model",
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate multi-horizon risk predictions separately for each horizon h in 1..H.

    Args:
        y_true: Ground truth binary attack indicators of shape (N, H) or (N,).
        y_probs: Predicted risk probabilities of shape (N, H) or (N,).
        model_name: Descriptive model identifier.
        threshold: Decision threshold for discrete classification (default: 0.5).

    Returns:
        Dictionary mapping 'horizon_1'..'horizon_H' to comprehensive metric dicts,
        plus 'overall_mean' summary.
    """
    y_t = np.asarray(y_true)
    y_p = _binary_probabilities(y_probs)

    if y_t.ndim == 1:
        y_t = y_t[:, np.newaxis]
    if y_p.ndim == 1:
        y_p = y_p[:, np.newaxis]

    if y_t.shape[0] != y_p.shape[0]:
        raise ValueError(f"Sample count mismatch: y_true has {y_t.shape[0]}, y_probs has {y_p.shape[0]}")

    h_steps = min(y_t.shape[1], y_p.shape[1])
    report: dict[str, Any] = {"model_name": model_name, "horizons": {}}

    f1_list: list[float] = []
    pr_auc_list: list[float] = []
    fpr_list: list[float] = []
    brier_list: list[float] = []

    for h in range(h_steps):
        t_col = (y_t[:, h] > 0.5).astype(int)
        p_col = y_p[:, h]

        n_samples = len(t_col)
        n_attack = int(np.sum(t_col == 1))
        n_benign = int(np.sum(t_col == 0))

        preds = (p_col >= threshold).astype(int)

        tp = int(np.sum((preds == 1) & (t_col == 1)))
        fp = int(np.sum((preds == 1) & (t_col == 0)))
        tn = int(np.sum((preds == 0) & (t_col == 0)))
        fn = int(np.sum((preds == 0) & (t_col == 1)))

        acc = float(accuracy_score(t_col, preds))
        prec = float(precision_score(t_col, preds, zero_division=0.0))
        rec = float(recall_score(t_col, preds, zero_division=0.0))
        f1 = float(f1_score(t_col, preds, zero_division=0.0))

        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        brier = float(np.mean((p_col - t_col) ** 2))

        # Precision-Recall AUC (Average Precision)
        if n_attack > 0 and n_benign > 0:
            pr_auc = float(average_precision_score(t_col, p_col))
            roc_auc = float(roc_auc_score(t_col, p_col))
        elif n_attack == n_samples:
            pr_auc = 1.0
            roc_auc = 1.0
        else:
            pr_auc = 0.0
            roc_auc = 0.0

        # Find threshold that maximizes F1 on this partition
        best_th = 0.5
        best_f1 = f1
        for candidate_th in np.linspace(0.1, 0.9, 17):
            cand_preds = (p_col >= candidate_th).astype(int)
            cand_f1 = float(f1_score(t_col, cand_preds, zero_division=0.0))
            if cand_f1 > best_f1:
                best_f1 = cand_f1
                best_th = float(candidate_th)

        h_key = f"horizon_{h + 1}"
        h_metrics = {
            "horizon_step": h + 1,
            "samples": n_samples,
            "benign_count": n_benign,
            "attack_count": n_attack,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "fpr": fpr,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "brier_score": brier,
            "optimal_threshold": best_th,
            "optimal_f1": best_f1,
            "precision_undefined": bool(tp + fp == 0),
        }
        report["horizons"][h_key] = h_metrics
        report[h_key] = h_metrics  # Top-level alias for convenient access

        f1_list.append(f1)
        pr_auc_list.append(pr_auc)
        fpr_list.append(fpr)
        brier_list.append(brier)

    report["overall_mean"] = {
        "mean_f1": float(np.mean(f1_list)) if f1_list else 0.0,
        "mean_pr_auc": float(np.mean(pr_auc_list)) if pr_auc_list else 0.0,
        "mean_fpr": float(np.mean(fpr_list)) if fpr_list else 0.0,
        "mean_brier": float(np.mean(brier_list)) if brier_list else 0.0,
    }
    return report


def evaluate_state_reconstruction(
    X_true: np.ndarray,
    X_pred: np.ndarray,
) -> dict[str, Any]:
    """Evaluate continuous network state rollout fidelity across horizons.

    Args:
        X_true: Ground truth standardized flow features of shape (N, H, D).
        X_pred: Model-predicted standardized flow features of shape (N, H, D).

    Returns:
        Dictionary of MSE, MAE, and Cosine Similarity per horizon step and overall.
    """
    x_t = np.asarray(X_true, dtype=np.float32)
    x_p = np.asarray(X_pred, dtype=np.float32)

    if x_t.shape != x_p.shape:
        raise ValueError(f"Shape mismatch: X_true has {x_t.shape}, X_pred has {x_p.shape}")

    n_samples, h_steps, n_features = x_t.shape
    per_horizon = {}

    for h in range(h_steps):
        true_h = x_t[:, h, :]
        pred_h = x_p[:, h, :]

        mse = float(np.mean((true_h - pred_h) ** 2))
        mae = float(np.mean(np.abs(true_h - pred_h)))

        # Cosine similarity per sample
        norm_t = np.linalg.norm(true_h, axis=1, keepdims=True)
        norm_p = np.linalg.norm(pred_h, axis=1, keepdims=True)
        denom = norm_t * norm_p
        denom[denom < 1e-8] = 1.0
        cos_sim = float(np.mean(np.sum(true_h * pred_h, axis=1, keepdims=True) / denom))

        per_horizon[f"horizon_{h + 1}"] = {
            "step": h + 1,
            "mse": mse,
            "mae": mae,
            "cosine_similarity": cos_sim,
        }

    overall_mse = float(np.mean((x_t - x_p) ** 2))
    overall_mae = float(np.mean(np.abs(x_t - x_p)))

    return {
        "overall_mse": overall_mse,
        "overall_mae": overall_mae,
        "per_horizon": per_horizon,
    }


def compare_all_models(
    world_model_probs: np.ndarray,
    lr_probs: np.ndarray,
    rf_probs: np.ndarray,
    y_true: np.ndarray,
    state_metrics: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Generate consolidated side-by-side comparison across all 4 forecast horizons."""
    y_t = np.asarray(y_true)
    wm_res = evaluate_model_predictions(y_t, world_model_probs, model_name="Temporal World Model")
    lr_res = evaluate_model_predictions(y_t, lr_probs, model_name="Logistic Regression")
    rf_res = evaluate_model_predictions(y_t, rf_probs, model_name="Random Forest")

    rows = []
    models = [
        ("Temporal World Model", wm_res),
        ("Random Forest", rf_res),
        ("Logistic Regression", lr_res),
    ]

    for model_name, res in models:
        for h in range(1, FORECAST_STEPS + 1):
            h_key = f"horizon_{h}"
            if h_key in res:
                h_data = res[h_key]
                rows.append({
                    "Model": model_name,
                    "Horizon": f"+{h} Flow(s)",
                    "F1-Score": h_data["f1"],
                    "Precision": h_data["precision"],
                    "Recall": h_data["recall"],
                    "PR-AUC": h_data["pr_auc"],
                    "FPR": h_data["fpr"],
                    "Brier Score": h_data["brier_score"],
                })

    df_summary = pd.DataFrame(rows)
    all_results = {
        "world_model": wm_res,
        "random_forest": rf_res,
        "logistic_regression": lr_res,
        "state_reconstruction": state_metrics,
        "summary_table": rows,
    }
    return all_results, df_summary
