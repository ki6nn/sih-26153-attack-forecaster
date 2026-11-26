"""Reproducible classical baselines for multi-horizon attack forecasting.

Provides:
    - LogisticRegressionBaseline: L2-regularized linear baseline over flattened windows (200 dims).
    - RandomForestBaseline: Multi-horizon decision tree ensemble over summary statistics (100 dims).
    - Unified 4-horizon prediction interface (predict_risk_proba returning shape (N, 4)).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.contracts import (
    ATTACK_CATEGORY_GROUPS,
    FORECAST_STEPS,
    IGNORE_CATEGORY_INDEX,
    NUM_ATTACK_CATEGORIES,
    NUM_FEATURES,
    WINDOW_SIZE,
)


def _prepare_risk_targets(y_risk: Any, n_samples: int, forecast_steps: int = FORECAST_STEPS) -> np.ndarray:
    """Standardize risk targets into a 2D float32 array of shape (N, H)."""
    arr = np.asarray(y_risk, dtype=np.float32)
    if arr.ndim == 1:
        if len(arr) != n_samples:
            raise ValueError(f"Sample count mismatch: expected {n_samples}, got {len(arr)}")
        # Tile across all forecast horizons
        return np.tile(arr[:, np.newaxis], (1, forecast_steps))
    if arr.ndim == 2:
        if arr.shape != (n_samples, forecast_steps):
            raise ValueError(f"Expected shape ({n_samples}, {forecast_steps}), got {arr.shape}")
        return arr
    raise ValueError(f"Risk target must be 1D or 2D array, got shape {arr.shape}")


def _encode_category_labels(y_category: Any, n_samples: int) -> np.ndarray:
    """Encode category group labels into integer indices, masking unobserved labels with -100."""
    arr = np.asarray(y_category)
    if arr.ndim == 2:
        # Use immediate step 1 category for baseline classification
        arr = arr[:, 0]
    arr = arr.reshape(-1)
    if len(arr) != n_samples:
        raise ValueError(f"Category sample count mismatch: expected {n_samples}, got {len(arr)}")

    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.int64)

    lookup = {name.lower(): idx for idx, name in enumerate(ATTACK_CATEGORY_GROUPS)}
    encoded = []
    for item in arr:
        val = str(item).strip().lower()
        idx = lookup.get(val, IGNORE_CATEGORY_INDEX)
        encoded.append(idx)
    return np.asarray(encoded, dtype=np.int64)


class LogisticRegressionBaseline:
    """Multi-horizon L2-regularized linear baseline over flattened input windows.

    Representation:
        Flattens (W=10, D=20) historical sequence into a single 200-dimensional vector
        capturing linear cross-temporal and cross-feature interactions.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
        forecast_steps: int = FORECAST_STEPS,
        save_path: str = "artifacts/baseline_lr.pkl",
    ) -> None:
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.forecast_steps = forecast_steps
        self.save_path = save_path

        # Independent binary classifier per horizon step
        self.risk_models = [
            LogisticRegression(
                C=C,
                max_iter=max_iter,
                random_state=random_state + h,
                solver="lbfgs",
            )
            for h in range(forecast_steps)
        ]
        self.category_model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            solver="lbfgs",
        )
        self.is_fitted = False
        self.category_names = list(ATTACK_CATEGORY_GROUPS)

    def _flatten_input(self, X: Any) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 3:
            if arr.shape[1:] != (WINDOW_SIZE, NUM_FEATURES):
                raise ValueError(
                    f"Expected shape (N, {WINDOW_SIZE}, {NUM_FEATURES}), got {arr.shape}"
                )
            return arr.reshape(arr.shape[0], -1)
        if arr.ndim == 2:
            if arr.shape[1] == (WINDOW_SIZE * NUM_FEATURES):
                return arr
            if arr.shape[1] == NUM_FEATURES:
                # 2D fallback: tile across W to produce compatible dimensions
                return np.tile(arr, (1, WINDOW_SIZE))
        raise ValueError(
            f"Input features must have shape (N, {WINDOW_SIZE}, {NUM_FEATURES}) or (N, {WINDOW_SIZE * NUM_FEATURES})"
        )

    def fit(
        self,
        X_train: Any,
        y_risk: Any,
        y_category: Any = None,
        save_path: str | None = None,
    ) -> "LogisticRegressionBaseline":
        X_flat = self._flatten_input(X_train)
        n_samples = X_flat.shape[0]
        y_r = _prepare_risk_targets(y_risk, n_samples, self.forecast_steps)

        # Fit independent classifier for each horizon step
        for h in range(self.forecast_steps):
            col_target = (y_r[:, h] > 0.5).astype(int)
            unique_classes = np.unique(col_target)
            if len(unique_classes) > 1:
                self.risk_models[h].fit(X_flat, col_target)
            else:
                # Single-class edge case: mock fitted model returning constant probability
                self.risk_models[h].classes_ = unique_classes

        if y_category is not None:
            cat_labels = _encode_category_labels(y_category, n_samples)
            valid_mask = cat_labels != IGNORE_CATEGORY_INDEX
            if valid_mask.any() and len(np.unique(cat_labels[valid_mask])) > 1:
                self.category_model.fit(X_flat[valid_mask], cat_labels[valid_mask])
            elif valid_mask.any():
                self.category_model.classes_ = np.unique(cat_labels[valid_mask])

        self.is_fitted = True
        dest = Path(save_path or self.save_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, dest)
        return self

    def predict_risk_proba(self, X: Any) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("LogisticRegressionBaseline is not fitted yet.")
        X_flat = self._flatten_input(X)
        n_samples = X_flat.shape[0]
        probs = np.zeros((n_samples, self.forecast_steps), dtype=np.float32)

        for h in range(self.forecast_steps):
            model = self.risk_models[h]
            if hasattr(model, "predict_proba"):
                if len(model.classes_) == 2:
                    probs[:, h] = model.predict_proba(X_flat)[:, 1]
                else:
                    probs[:, h] = float(model.classes_[0])
            else:
                probs[:, h] = 0.0
        return probs

    def predict_category(self, X: Any) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("LogisticRegressionBaseline is not fitted yet.")
        X_flat = self._flatten_input(X)
        if hasattr(self.category_model, "predict"):
            preds = self.category_model.predict(X_flat)
            return np.asarray([
                self.category_names[i] if 0 <= i < len(self.category_names) else "Normal"
                for i in preds
            ])
        return np.asarray(["Normal"] * X_flat.shape[0])

    def predict_proba(self, X: Any) -> np.ndarray:
        """Alias for predict_risk_proba providing standard scikit-learn compatibility."""
        return self.predict_risk_proba(X)

    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        """Predict discrete binary attack indicators across all forecast horizons."""
        return (self.predict_risk_proba(X) >= threshold).astype(int)

    def save(self, save_path: str | None = None) -> str:
        dest = Path(save_path or self.save_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, dest)
        return str(dest)

    @classmethod
    def load(cls, save_path: str = "artifacts/baseline_lr.pkl") -> "LogisticRegressionBaseline":
        return joblib.load(save_path)


class RandomForestBaseline:
    """Multi-horizon Random Forest ensemble baseline over temporal summary statistics.

    Representation:
        Extracts the latest flow features (20 dims) concatenated with statistical summary
        features (mean, std, min, max across the 10 timesteps -> 80 dims) yielding 100 features.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 12,
        random_state: int = 42,
        forecast_steps: int = FORECAST_STEPS,
        save_path: str = "artifacts/baseline_rf.pkl",
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.forecast_steps = forecast_steps
        self.save_path = save_path

        self.risk_models = [
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state + h,
                class_weight="balanced",
                n_jobs=-1,
            )
            for h in range(forecast_steps)
        ]
        self.category_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
        self.is_fitted = False
        self.category_names = list(ATTACK_CATEGORY_GROUPS)

    def _extract_summary_features(self, X: Any) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 3:
            if arr.shape[1:] != (WINDOW_SIZE, NUM_FEATURES):
                raise ValueError(
                    f"Expected shape (N, {WINDOW_SIZE}, {NUM_FEATURES}), got {arr.shape}"
                )
            latest = arr[:, -1, :]  # Shape: (N, 20)
            mean_f = np.mean(arr, axis=1)  # (N, 20)
            std_f = np.std(arr, axis=1)  # (N, 20)
            min_f = np.min(arr, axis=1)  # (N, 20)
            max_f = np.max(arr, axis=1)  # (N, 20)
            return np.concatenate([latest, mean_f, std_f, min_f, max_f], axis=1)  # Shape: (N, 100)
        if arr.ndim == 2 and arr.shape[1] == NUM_FEATURES:
            # 2D fallback: tile 5 times to produce 100 dimensions
            return np.tile(arr, (1, 5))
        if arr.ndim == 2 and arr.shape[1] == 100:
            return arr
        raise ValueError(
            f"Input features must have shape (N, {WINDOW_SIZE}, {NUM_FEATURES}) or (N, 100), got {arr.shape}"
        )

    def fit(
        self,
        X_train: Any,
        y_risk: Any,
        y_category: Any = None,
        save_path: str | None = None,
    ) -> "RandomForestBaseline":
        X_feats = self._extract_summary_features(X_train)
        n_samples = X_feats.shape[0]
        y_r = _prepare_risk_targets(y_risk, n_samples, self.forecast_steps)

        for h in range(self.forecast_steps):
            col_target = (y_r[:, h] > 0.5).astype(int)
            unique_classes = np.unique(col_target)
            if len(unique_classes) > 1:
                self.risk_models[h].fit(X_feats, col_target)
            else:
                self.risk_models[h].classes_ = unique_classes

        if y_category is not None:
            cat_labels = _encode_category_labels(y_category, n_samples)
            valid_mask = cat_labels != IGNORE_CATEGORY_INDEX
            if valid_mask.any() and len(np.unique(cat_labels[valid_mask])) > 1:
                self.category_model.fit(X_feats[valid_mask], cat_labels[valid_mask])
            elif valid_mask.any():
                self.category_model.classes_ = np.unique(cat_labels[valid_mask])

        self.is_fitted = True
        dest = Path(save_path or self.save_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, dest)
        return self

    def predict_risk_proba(self, X: Any) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("RandomForestBaseline is not fitted yet.")
        X_feats = self._extract_summary_features(X)
        n_samples = X_feats.shape[0]
        probs = np.zeros((n_samples, self.forecast_steps), dtype=np.float32)

        for h in range(self.forecast_steps):
            model = self.risk_models[h]
            if hasattr(model, "predict_proba"):
                if len(model.classes_) == 2:
                    probs[:, h] = model.predict_proba(X_feats)[:, 1]
                else:
                    probs[:, h] = float(model.classes_[0])
            else:
                probs[:, h] = 0.0
        return probs

    def predict_category(self, X: Any) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("RandomForestBaseline is not fitted yet.")
        X_feats = self._extract_summary_features(X)
        if hasattr(self.category_model, "predict"):
            preds = self.category_model.predict(X_feats)
            return np.asarray([
                self.category_names[i] if 0 <= i < len(self.category_names) else "Normal"
                for i in preds
            ])
        return np.asarray(["Normal"] * X_feats.shape[0])

    def predict_proba(self, X: Any) -> np.ndarray:
        """Alias for predict_risk_proba providing standard scikit-learn compatibility."""
        return self.predict_risk_proba(X)

    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        """Predict discrete binary attack indicators across all forecast horizons."""
        return (self.predict_risk_proba(X) >= threshold).astype(int)

    def save(self, save_path: str | None = None) -> str:
        dest = Path(save_path or self.save_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, dest)
        return str(dest)

    @classmethod
    def load(cls, save_path: str = "artifacts/baseline_rf.pkl") -> "RandomForestBaseline":
        return joblib.load(save_path)


# Backward compatibility adapter for legacy BaselineAttackClassifier callers
class BaselineAttackClassifier(RandomForestBaseline):
    """Adapter maintaining backward compatibility with legacy BaselineAttackClassifier."""

    def __init__(
        self,
        n_estimators: int = 100,
        random_state: int = 42,
        save_path: str = "artifacts/baseline_rf.pkl",
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            random_state=random_state,
            save_path=save_path,
        )

    def predict_attack(self, features) -> np.ndarray:
        probs = self.predict_risk_proba(features)
        return (probs[:, 0] >= 0.5).astype(int)

    def predict_attack_proba(self, features) -> np.ndarray:
        probs = self.predict_risk_proba(features)
        return probs[:, 0]

    def predict_stage(self, features) -> np.ndarray:
        return self.predict_category(features)


def evaluate_models(
    world_model_probs,
    baseline_probs,
    y_true,
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    """Legacy evaluation adapter comparing step 1 probabilities."""
    from src.model.evaluation import evaluate_model_predictions

    y_t = np.asarray(y_true).reshape(-1)
    w_p = np.asarray(world_model_probs)
    b_p = np.asarray(baseline_probs)

    w_res = evaluate_model_predictions(y_t, w_p, model_name="World Model")
    b_res = evaluate_model_predictions(y_t, b_p, model_name="Baseline RF")

    # Extract step 1 comparison
    comp = {
        "World Model": {
            "Accuracy": w_res["horizon_1"]["accuracy"],
            "Precision": w_res["horizon_1"]["precision"],
            "Recall": w_res["horizon_1"]["recall"],
            "F1-Score": w_res["horizon_1"]["f1"],
            "ROC-AUC": w_res["horizon_1"]["roc_auc"],
        },
        "Baseline RF": {
            "Accuracy": b_res["horizon_1"]["accuracy"],
            "Precision": b_res["horizon_1"]["precision"],
            "Recall": b_res["horizon_1"]["recall"],
            "F1-Score": b_res["horizon_1"]["f1"],
            "ROC-AUC": b_res["horizon_1"]["roc_auc"],
        },
    }
    df = pd.DataFrame(comp).T
    df.index.name = "Model"
    return comp, df
