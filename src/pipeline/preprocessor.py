"""Training-only preprocessing module with strict isolation and artifact serialization."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Sequence

import joblib
import numpy as np
import pandas as pd

from src.contracts import (
    FORECAST_STEPS,
    MITRE_STAGE_MAP,
    NUM_FEATURES,
    WINDOW_SIZE,
)
from src.pipeline.features import (
    UNSW_NB15_FEATURES,
    extract_features,
    map_stage_labels,
)

FEATURE_COLUMNS: Final[list[str]] = list(UNSW_NB15_FEATURES)


class NetworkDataPreprocessor:
    """Scaler and preprocessor fitted strictly on training data to prevent leakage.

    Provides mathematical equivalence to StandardScaler or RobustScaler without
    requiring platform-specific C-extensions (fully compatible with Windows WDAC).
    """

    def __init__(
        self,
        method: str = "standard",
        feature_columns: Sequence[str] | None = None,
        clip_range: tuple[float, float] | None = (-10.0, 10.0),
    ) -> None:
        self.method = method.lower()
        self.feature_columns = list(feature_columns) if feature_columns else list(FEATURE_COLUMNS)
        self.clip_range = clip_range
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.is_fitted: bool = False
        self.metadata_: dict[str, object] = {}

    def fit(self, X: pd.DataFrame | np.ndarray) -> "NetworkDataPreprocessor":
        """Fit center and scale parameters strictly on training partition."""
        data = self._to_numpy(X)
        if data.ndim != 2:
            raise ValueError(f"Expected 2D input array, got shape {data.shape}")
        if data.shape[1] != len(self.feature_columns):
            raise ValueError(
                f"Expected {len(self.feature_columns)} features, got {data.shape[1]}"
            )

        if self.method == "standard":
            center = np.mean(data, axis=0)
            scale = np.std(data, axis=0)
            # Avoid division by zero for constant features
            scale[scale < 1e-6] = 1.0
        elif self.method == "robust":
            center = np.median(data, axis=0)
            q75 = np.percentile(data, 75, axis=0)
            q25 = np.percentile(data, 25, axis=0)
            scale = q75 - q25
            scale[scale < 1e-6] = 1.0
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")

        self.center_ = center.astype(np.float32)
        self.scale_ = scale.astype(np.float32)
        self.is_fitted = True
        self.metadata_ = {
            "method": self.method,
            "num_features": len(self.feature_columns),
            "feature_columns": self.feature_columns,
            "sample_count": int(data.shape[0]),
        }
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Apply pre-computed scaling parameters without refitting."""
        if not self.is_fitted or self.center_ is None or self.scale_ is None:
            raise RuntimeError("Preprocessor has not been fitted yet. Call fit() first.")

        data = self._to_numpy(X)
        if data.ndim != 2 or data.shape[1] != len(self.feature_columns):
            raise ValueError(
                f"Feature mismatch: expected shape (N, {len(self.feature_columns)}), got {data.shape}"
            )

        scaled = (data - self.center_) / self.scale_
        if self.clip_range is not None:
            scaled = np.clip(scaled, self.clip_range[0], self.clip_range[1])
        return scaled.astype(np.float32)

    def fit_transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Fit on training data and return scaled array."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse scaling transformation."""
        if not self.is_fitted or self.center_ is None or self.scale_ is None:
            raise RuntimeError("Preprocessor has not been fitted yet.")
        return (X * self.scale_) + self.center_

    def save(self, filepath: str | Path = "artifacts/preprocessor.joblib") -> str:
        """Serialize fitted preprocessor to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted preprocessor.")
        dest = Path(filepath)
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, dest)
        return str(dest)

    @staticmethod
    def load(filepath: str | Path = "artifacts/preprocessor.joblib") -> "NetworkDataPreprocessor":
        """Load serialized preprocessor."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at {path}")
        obj = joblib.load(path)
        if not isinstance(obj, NetworkDataPreprocessor):
            raise TypeError(f"Loaded object is of type {type(obj)}, expected NetworkDataPreprocessor")
        return obj

    def _to_numpy(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            # Select target columns in order if present
            cols = [c for c in self.feature_columns if c in X.columns]
            if len(cols) == len(self.feature_columns):
                arr = X[self.feature_columns].to_numpy(dtype=np.float32)
            else:
                arr = X.to_numpy(dtype=np.float32)
        else:
            arr = np.asarray(X, dtype=np.float32)
        # Handle remaining NaNs/Infs
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
        return arr


def preprocess(
    data: pd.DataFrame | None = None,
    dataset_name: str = "unsw_nb15",
    preprocessor: NetworkDataPreprocessor | None = None,
    scaler_path: str | Path = "artifacts/scaler.pkl",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preprocess DataFrame and return (X, y_probs, y_stage) temporal arrays.

    Note: When calling preprocess() on raw full datasets, fit the preprocessor
    strictly on the training split before transforming validation/test sets.
    """
    if data is None:
        data_path = Path("data/sample_demo.csv")
        if not data_path.exists():
            raise FileNotFoundError(
                "No input data provided and data/sample_demo.csv does not exist. "
                "Download or provide official dataset files before training."
            )
        data = pd.read_csv(data_path)

    features = extract_features(data, dataset_name=dataset_name)

    # Determine stage and attack targets
    if "attack_cat" in data.columns:
        stages = map_stage_labels(data["attack_cat"], dataset_name=dataset_name).to_numpy()
    elif "label" in data.columns:
        stages = np.where(data["label"].to_numpy() == 0, "Normal", "Generic")
    else:
        stages = np.array(["Normal"] * len(data))

    if preprocessor is None:
        preprocessor = NetworkDataPreprocessor(feature_columns=list(features.columns))
        scaled = preprocessor.fit_transform(features)
        preprocessor.save(scaler_path)
    else:
        scaled = preprocessor.transform(features)

    sample_count = len(data) - WINDOW_SIZE - FORECAST_STEPS + 1
    if sample_count <= 0:
        raise ValueError(
            f"At least {WINDOW_SIZE + FORECAST_STEPS} rows required for windowing; got {len(data)}"
        )

    X = np.stack([scaled[i : i + WINDOW_SIZE] for i in range(sample_count)])
    future_stages = np.stack(
        [stages[i + WINDOW_SIZE : i + WINDOW_SIZE + FORECAST_STEPS] for i in range(sample_count)]
    )
    y_probs = (future_stages != "Normal").astype(np.float32)

    stage_indices = {stage: index for index, stage in enumerate(MITRE_STAGE_MAP)}
    y_stage = np.asarray(
        [stage_indices.get(future[0], 0) for future in future_stages], dtype=np.int64
    )
    return X, y_probs, y_stage
