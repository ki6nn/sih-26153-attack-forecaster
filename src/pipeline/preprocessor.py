"""Training-only preprocessing module with strict isolation and artifact serialization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Sequence

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
    get_feature_columns,
    map_stage_labels,
)
from src.pipeline.ingestion import load_dataset
from src.pipeline.validation import validate_required_columns
from src.pipeline.windowing import chronological_split, create_network_windows

FEATURE_COLUMNS: Final[list[str]] = list(UNSW_NB15_FEATURES)


@dataclass
class ResearchPipelineData:
    """Encapsulates partitioned, leak-free temporal data ready for model training/evaluation."""

    X_train: np.ndarray
    y_probs_train: np.ndarray
    y_stage_train: np.ndarray
    X_val: np.ndarray
    y_probs_val: np.ndarray
    y_stage_val: np.ndarray
    X_test: np.ndarray
    y_probs_test: np.ndarray
    y_stage_test: np.ndarray
    preprocessor: NetworkDataPreprocessor
    feature_names: list[str]
    isolation_audit: dict[str, Any]
    intervals_train: pd.DataFrame
    intervals_val: pd.DataFrame
    intervals_test: pd.DataFrame


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
            cols = [c for c in self.feature_columns if c in X.columns]
            if len(cols) == len(self.feature_columns):
                arr = X[self.feature_columns].to_numpy(dtype=np.float32)
            else:
                arr = X.to_numpy(dtype=np.float32)
        else:
            arr = np.asarray(X, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
        return arr


def _extract_partition_targets(
    partition_df: pd.DataFrame, dataset_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Helper to extract binary attack labels and integer stage labels from a partition."""
    if "attack_cat" in partition_df.columns:
        stages_mapped = map_stage_labels(partition_df["attack_cat"], dataset_name=dataset_name).to_numpy()
        y_attack = (stages_mapped != "Normal").astype(int)
    elif "label" in partition_df.columns:
        raw_label = pd.to_numeric(partition_df["label"], errors="coerce").fillna(0).astype(int).to_numpy()
        y_attack = (raw_label > 0).astype(int)
        stages_mapped = np.where(y_attack == 0, "Normal", "Generic")
    else:
        y_attack = np.zeros(len(partition_df), dtype=int)
        stages_mapped = np.array(["Normal"] * len(partition_df))

    stage_lookup = {stg: idx for idx, stg in enumerate(MITRE_STAGE_MAP)}
    y_stage = np.asarray([stage_lookup.get(stg, 0) for stg in stages_mapped], dtype=np.int64)
    return y_attack, y_stage


def prepare_research_pipeline(
    data_or_path: str | Path | pd.DataFrame,
    dataset_name: str = "unsw_nb15",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    timestamp_col: str | None = None,
    duration_col: str | None = None,
    duration_unit: str = "s",
    min_gap_seconds: float = 0.0,
    preprocessor_method: str = "standard",
    scaler_save_path: str | Path = "artifacts/scaler.pkl",
) -> ResearchPipelineData:
    """Production research pipeline: zero data leakage, chronological-first partitioning,

    training-only scaler fitting, and partition-isolated temporal window generation.

    Execution Flow:
        1. Ingest and validate DataFrame.
        2. Enforce 20-feature dataset contract.
        3. Split chronologically into Train, Validation, and Test partitions first.
        4. Fit NetworkDataPreprocessor strictly on Training features only.
        5. Transform Validation and Test features out-of-sample using the fitted preprocessor.
        6. Construct sliding windows independently within each partition boundary.
        7. Audit and verify temporal isolation before returning.

    Raises:
        ValueError: If timestamps or durations are invalid or temporal isolation fails.
    """
    # 1. Ingest dataset
    if isinstance(data_or_path, (str, Path)):
        df = load_dataset(data_or_path, dataset_name=dataset_name)
    else:
        df = data_or_path.copy()

    if df.empty:
        raise ValueError("Cannot prepare research pipeline on an empty DataFrame.")

    # 2. Validate feature schema
    feature_cols = get_feature_columns(dataset_name)
    validate_required_columns(df, feature_cols)

    # Auto-detect default duration columns if present
    dur_col = duration_col
    if dur_col is None:
        if "dur" in df.columns:
            dur_col = "dur"
            duration_unit = "s"
        elif "flow_duration" in df.columns:
            dur_col = "flow_duration"
            duration_unit = "us"

    # 3. Chronological partitioning FIRST
    train_df, val_df, test_df = chronological_split(
        df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        timestamp_col=timestamp_col,
        duration_col=dur_col,
        duration_unit=duration_unit,
        min_gap_seconds=min_gap_seconds,
        allow_index_fallback=False,  # Strictly forbidden for research
    )

    # 4. Feature extraction per partition
    train_feat = extract_features(train_df, dataset_name=dataset_name, feature_columns=feature_cols)
    val_feat = extract_features(val_df, dataset_name=dataset_name, feature_columns=feature_cols)
    test_feat = extract_features(test_df, dataset_name=dataset_name, feature_columns=feature_cols)

    # 5. Preprocessor fitted ONLY on training partition
    preprocessor = NetworkDataPreprocessor(method=preprocessor_method, feature_columns=feature_cols)
    scaled_train = preprocessor.fit_transform(train_feat)
    preprocessor.save(scaler_save_path)

    # 6. Transform validation and test out-of-sample
    scaled_val = preprocessor.transform(val_feat)
    scaled_test = preprocessor.transform(test_feat)

    # Extract labels
    y_att_train, y_stg_train = _extract_partition_targets(train_df, dataset_name)
    y_att_val, y_stg_val = _extract_partition_targets(val_df, dataset_name)
    y_att_test, y_stg_test = _extract_partition_targets(test_df, dataset_name)

    # Identify timestamp array for interval audits
    ts_col = timestamp_col or ("datetime" if "datetime" in df.columns else None)
    train_ts = train_df[ts_col].to_numpy() if ts_col and ts_col in train_df.columns else None
    val_ts = val_df[ts_col].to_numpy() if ts_col and ts_col in val_df.columns else None
    test_ts = test_df[ts_col].to_numpy() if ts_col and ts_col in test_df.columns else None

    # 7. Partition-isolated window generation
    X_train_w, y_prob_train_w, y_stg_train_w, int_train = create_network_windows(
        scaled_train, y_att_train, y_stg_train, timestamps=train_ts, return_intervals=True
    )
    X_val_w, y_prob_val_w, y_stg_val_w, int_val = create_network_windows(
        scaled_val, y_att_val, y_stg_val, timestamps=val_ts, return_intervals=True
    )
    X_test_w, y_prob_test_w, y_stg_test_w, int_test = create_network_windows(
        scaled_test, y_att_test, y_stg_test, timestamps=test_ts, return_intervals=True
    )

    # Isolation audit
    isolation_audit: dict[str, Any] = {}
    if ts_col and ts_col in df.columns:
        from src.pipeline.windowing import verify_temporal_isolation

        isolation_audit = verify_temporal_isolation(
            train_df,
            val_df,
            test_df,
            timestamp_col=ts_col,
            duration_col=dur_col,
            duration_unit=duration_unit,
        )

    return ResearchPipelineData(
        X_train=X_train_w,
        y_probs_train=y_prob_train_w,
        y_stage_train=y_stg_train_w,
        X_val=X_val_w,
        y_probs_val=y_prob_val_w,
        y_stage_val=y_stg_val_w,
        X_test=X_test_w,
        y_probs_test=y_prob_test_w,
        y_stage_test=y_stg_test_w,
        preprocessor=preprocessor,
        feature_names=feature_cols,
        isolation_audit=isolation_audit,
        intervals_train=int_train,
        intervals_val=int_val,
        intervals_test=int_test,
    )


def preprocess(
    data: pd.DataFrame | None = None,
    dataset_name: str = "unsw_nb15",
    preprocessor: NetworkDataPreprocessor | None = None,
    scaler_path: str | Path = "artifacts/scaler.pkl",
    is_research: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convenience preprocessor for offline SOC dashboard UI demonstrations.

    STRICT POLICY:
        - If is_research=True, this function raises ValueError. Use prepare_research_pipeline()
          for all empirical research and benchmarking to prevent full-data scaling leakage.
        - If data=None, loads data/sample_demo.csv strictly as non-research synthetic demonstration data.
    """
    if is_research:
        raise ValueError(
            "Full-dataset preprocess() is strictly prohibited for research workflows because "
            "it fits transformations across the entire dataset. Call prepare_research_pipeline() instead."
        )

    if data is None:
        data_path = Path("data/sample_demo.csv")
        if not data_path.exists():
            raise FileNotFoundError(
                "No input data provided and data/sample_demo.csv does not exist. "
                "Download or provide official dataset files before training."
            )
        data = pd.read_csv(data_path)

    features = extract_features(data, dataset_name=dataset_name)

    if "attack_cat" in data.columns:
        stages = map_stage_labels(data["attack_cat"], dataset_name=dataset_name).to_numpy()
    elif "label" in data.columns:
        stages = np.where(data["label"].to_numpy() == 0, "Normal", "Generic")
    else:
        stages = np.array(["Normal"] * len(data))

    if preprocessor is None:
        preprocessor = NetworkDataPreprocessor(feature_columns=list(features.columns))
        scaled = preprocessor.fit_transform(features)
        preprocessor.metadata_["is_research_evidence"] = False
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
