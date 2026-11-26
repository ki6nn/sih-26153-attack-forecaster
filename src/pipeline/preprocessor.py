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
    detect_feature_anomalies,
    extract_features,
    get_feature_columns,
    map_stage_labels,
)
from src.pipeline.ingestion import (
    DATASET_SPECS,
    DatasetProvenance,
    load_dataset,
    verify_dataset_provenance,
)
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
    Enforces training-only imputation: missing/infinite values are imputed using
    statistics calculated exclusively from the training partition.
    """

    def __init__(
        self,
        method: str = "standard",
        feature_columns: Sequence[str] | None = None,
        clip_range: tuple[float, float] | None = (-10.0, 10.0),
        impute_strategy: str | None = "median",
    ) -> None:
        self.method = method.lower()
        self.feature_columns = list(feature_columns) if feature_columns else list(FEATURE_COLUMNS)
        self.clip_range = clip_range
        self.impute_strategy = impute_strategy.lower() if impute_strategy else None
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.impute_values_: dict[str, float] = {}
        self.is_fitted: bool = False
        self.metadata_: dict[str, object] = {}

    def fit(self, X: pd.DataFrame | np.ndarray) -> "NetworkDataPreprocessor":
        """Fit center, scale, and imputation parameters strictly on training partition."""
        data = self._to_numpy(X)
        if data.ndim != 2:
            raise ValueError(f"Expected 2D input array, got shape {data.shape}")
        if data.shape[1] != len(self.feature_columns):
            raise ValueError(
                f"Expected {len(self.feature_columns)} features, got {data.shape[1]}"
            )

        # 1. Fit training-only imputation values
        self.impute_values_ = {}
        imputed_data = np.copy(data)
        for idx, col in enumerate(self.feature_columns):
            col_vals = data[:, idx]
            finite_mask = np.isfinite(col_vals)
            finite_vals = col_vals[finite_mask]

            if not finite_mask.all() and self.impute_strategy is None:
                raise ValueError(
                    f"Feature '{col}' contains non-finite values (NaN/Inf) and impute_strategy is None."
                )

            if len(finite_vals) == 0:
                fill_val = 0.0
            elif self.impute_strategy == "median":
                fill_val = float(np.median(finite_vals))
            elif self.impute_strategy == "mean":
                fill_val = float(np.mean(finite_vals))
            else:
                fill_val = 0.0

            self.impute_values_[col] = fill_val
            imputed_data[~finite_mask, idx] = fill_val

        # 2. Fit center and scale on imputed training partition
        if self.method == "standard":
            center = np.mean(imputed_data, axis=0)
            scale = np.std(imputed_data, axis=0)
            scale[scale < 1e-6] = 1.0
        elif self.method == "robust":
            center = np.median(imputed_data, axis=0)
            q75 = np.percentile(imputed_data, 75, axis=0)
            q25 = np.percentile(imputed_data, 25, axis=0)
            scale = q75 - q25
            scale[scale < 1e-6] = 1.0
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")

        self.center_ = center.astype(np.float32)
        self.scale_ = scale.astype(np.float32)
        self.is_fitted = True
        self.metadata_ = {
            "method": self.method,
            "impute_strategy": self.impute_strategy,
            "num_features": len(self.feature_columns),
            "feature_columns": self.feature_columns,
            "sample_count": int(data.shape[0]),
            "impute_values": dict(self.impute_values_),
        }
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Apply pre-computed training scaling and imputation parameters without refitting."""
        if not self.is_fitted or self.center_ is None or self.scale_ is None:
            raise RuntimeError("Preprocessor has not been fitted yet. Call fit() first.")

        data = self._to_numpy(X)
        if data.ndim != 2 or data.shape[1] != len(self.feature_columns):
            raise ValueError(
                f"Feature mismatch: expected shape (N, {len(self.feature_columns)}), got {data.shape}"
            )

        # Apply training-fitted imputation
        imputed_data = np.copy(data)
        for idx, col in enumerate(self.feature_columns):
            col_vals = imputed_data[:, idx]
            non_finite_mask = ~np.isfinite(col_vals)
            if non_finite_mask.any():
                fill_val = self.impute_values_.get(col, 0.0)
                imputed_data[non_finite_mask, idx] = fill_val

        scaled = (imputed_data - self.center_) / self.scale_
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
        return arr


def _extract_partition_targets(
    partition_df: pd.DataFrame,
    dataset_name: str,
    strict_labels: bool = True,
    allow_unknown_labels: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Helper to extract binary attack labels and integer stage labels from a partition.

    Policy:
        - Target label column is mandatory. If neither 'attack_cat' nor 'label' exists, raises ValueError.
        - Null, empty string, or NaN labels raise ValueError; target labels cannot silently become benign.
        - Invalid numeric values in 'label' (negative, non-binary) raise ValueError.
        - Unknown attack categories map to 'Generic' with attack=1 (or raise ValueError if strict).
        - MITRE stage mapping is a high-level heuristic categorization; it does NOT establish
          verified MITRE ATT&CK enterprise technique attributions.

    Raises:
        ValueError: If target labels are missing, null, or invalid.
    """
    if "attack_cat" not in partition_df.columns and "label" not in partition_df.columns:
        raise ValueError(
            f"Target label column missing in partition. Required one of ('attack_cat', 'label'), "
            f"found columns: {list(partition_df.columns[:10])}..."
        )

    if "attack_cat" in partition_df.columns:
        stages_mapped = map_stage_labels(
            partition_df["attack_cat"],
            dataset_name=dataset_name,
            strict=strict_labels,
            allow_unknown=allow_unknown_labels,
        ).to_numpy()
        y_attack = (stages_mapped != "Normal").astype(int)
    else:
        # Binary 'label' column
        raw_s = partition_df["label"]
        null_mask = raw_s.isna() | raw_s.astype(str).str.strip().str.lower().isin(["", "nan", "none", "null"])
        if null_mask.any():
            null_count = int(null_mask.sum())
            raise ValueError(
                f"Found {null_count} null or missing target labels in 'label' column. "
                f"Target labels must be fully observed."
            )
        try:
            numeric_label = pd.to_numeric(raw_s, errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Non-numeric values found in target label column 'label': {exc}") from exc

        invalid_mask = ~numeric_label.isin([0, 1, 0.0, 1.0])
        if invalid_mask.any():
            bad_vals = numeric_label[invalid_mask].unique().tolist()
            raise ValueError(
                f"Invalid binary target label values in 'label': {bad_vals}. "
                f"Expected binary 0 (benign) or 1 (attack)."
            )

        y_attack = numeric_label.astype(int).to_numpy()
        stages_mapped = np.where(y_attack == 0, "Normal", "Generic")

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
    duration_unit: str | None = None,
    feature_columns: Sequence[str] | None = None,
    min_gap_seconds: float = 0.0,
    preprocessor_method: str = "standard",
    impute_strategy: str = "median",
    scaler_save_path: str | Path = "artifacts/scaler.pkl",
    enforce_provenance: bool = True,
    provenance: DatasetProvenance | dict[str, Any] | str | Path | None = None,
    acknowledge_zero_duration: bool = False,
    allow_unknown_labels: bool = False,
) -> ResearchPipelineData:
    """Production research pipeline: zero data leakage, chronological-first partitioning,

    training-only scaler fitting, training-only imputation, and partition-isolated temporal window generation.

    Execution Flow:
        1. Ingest and validate DataFrame (verifying provenance if enforce_provenance=True).
        2. Enforce 20-feature dataset contract and configure dataset-specific temporal parameters.
        3. Split chronologically into Train, Validation, and Test partitions first.
        4. Fit NetworkDataPreprocessor strictly on Training features only (including imputation).
        5. Transform Validation and Test features out-of-sample using the fitted preprocessor.
        6. Validate target labels (fail closed on missing, null, or invalid labels).
        7. Construct sliding windows independently within each partition boundary.
        8. Audit and verify temporal isolation before returning.

    Raises:
        ValueError: If timestamps, durations, labels, provenance, or temporal isolation fails.
    """
    # 1. Ingest dataset and verify provenance
    if isinstance(data_or_path, (str, Path)):
        df = load_dataset(
            data_or_path,
            dataset_name=dataset_name,
            enforce_provenance=enforce_provenance,
            provenance=provenance,
            is_research=enforce_provenance,
            strict_timestamps=True,
        )
    else:
        df = data_or_path.copy()
        if enforce_provenance:
            prov_obj = provenance if provenance is not None else df.attrs.get("provenance")
            if prov_obj is None:
                raise ValueError(
                    "Research pipeline requires explicit dataset provenance. Pass a verified "
                    "DatasetProvenance record or set enforce_provenance=False for synthetic unit tests."
                )
            if isinstance(prov_obj, dict):
                prov_obj = DatasetProvenance.from_dict(prov_obj)
            if not getattr(prov_obj, "is_research_evidence", False) or getattr(prov_obj, "provenance_type", "") != "official_research":
                raise ValueError(
                    f"Dataset has provenance_type='{getattr(prov_obj, 'provenance_type', 'unknown')}' "
                    f"(is_research_evidence=False) and is strictly rejected for empirical research "
                    f"under SIH 26153 rules. Synthetic demonstration data cannot be used as research evidence."
                )

    if df.empty:
        raise ValueError("Cannot prepare research pipeline on an empty DataFrame.")

    # 2. Validate feature schema
    feature_cols = list(feature_columns) if feature_columns else get_feature_columns(dataset_name)
    validate_required_columns(df, feature_cols)

    # Resolve dataset-specific temporal configurations
    ts_col = timestamp_col
    dur_col = duration_col
    dur_unit = duration_unit

    if dataset_name in DATASET_SPECS:
        spec = DATASET_SPECS[dataset_name]
        if ts_col is None and spec.get("timestamp_col") in df.columns:
            ts_col = spec.get("timestamp_col")
        if dur_col is None and spec.get("duration_col") in df.columns:
            dur_col = spec.get("duration_col")
            if dur_unit is None:
                dur_unit = spec.get("duration_unit")
        elif dur_unit is None:
            dur_unit = spec.get("duration_unit")

    # Fallback to column discovery if not specified
    if ts_col is None:
        for cand in ["datetime", "timestamp", "stime"]:
            if cand in df.columns:
                ts_col = cand
                break

    if dur_col is None:
        for cand, default_u in [("dur", "s"), ("flow_duration", "us")]:
            if cand in df.columns:
                dur_col = cand
                if dur_unit is None:
                    dur_unit = default_u
                break

    if dur_unit is None:
        dur_unit = "s"

    if dur_col is None and not acknowledge_zero_duration:
        raise ValueError(
            f"No duration column found or specified for dataset '{dataset_name}'. In strict "
            f"research mode, flows cannot be assumed instantaneous (0-duration) without "
            f"explicit user acknowledgment. Pass duration_col or acknowledge_zero_duration=True."
        )

    # 3. Chronological partitioning FIRST
    train_df, val_df, test_df = chronological_split(
        df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        timestamp_col=ts_col,
        duration_col=dur_col,
        duration_unit=dur_unit,
        min_gap_seconds=min_gap_seconds,
        allow_index_fallback=False,  # Strictly forbidden for research
    )

    # 4. Feature extraction per partition & training-only imputation
    train_feat = extract_features(train_df, dataset_name=dataset_name, feature_columns=feature_cols)

    # 5. Preprocessor fitted ONLY on training partition
    preprocessor = NetworkDataPreprocessor(
        method=preprocessor_method,
        feature_columns=feature_cols,
        impute_strategy=impute_strategy,
    )
    scaled_train = preprocessor.fit_transform(train_feat)
    preprocessor.save(scaler_save_path)

    # 6. Transform validation and test out-of-sample using training imputation & scaling
    train_impute_vals = preprocessor.impute_values_
    val_feat = extract_features(
        val_df,
        dataset_name=dataset_name,
        feature_columns=feature_cols,
        impute_values=train_impute_vals,
    )
    test_feat = extract_features(
        test_df,
        dataset_name=dataset_name,
        feature_columns=feature_cols,
        impute_values=train_impute_vals,
    )
    scaled_val = preprocessor.transform(val_feat)
    scaled_test = preprocessor.transform(test_feat)

    # 7. Extract labels with fail-closed validation
    y_att_train, y_stg_train = _extract_partition_targets(
        train_df, dataset_name, strict_labels=True, allow_unknown_labels=allow_unknown_labels
    )
    y_att_val, y_stg_val = _extract_partition_targets(
        val_df, dataset_name, strict_labels=True, allow_unknown_labels=allow_unknown_labels
    )
    y_att_test, y_stg_test = _extract_partition_targets(
        test_df, dataset_name, strict_labels=True, allow_unknown_labels=allow_unknown_labels
    )

    # Identify timestamp array for interval audits
    active_ts = ts_col or ("datetime" if "datetime" in df.columns else None)
    train_ts = train_df[active_ts].to_numpy() if active_ts and active_ts in train_df.columns else None
    val_ts = val_df[active_ts].to_numpy() if active_ts and active_ts in val_df.columns else None
    test_ts = test_df[active_ts].to_numpy() if active_ts and active_ts in test_df.columns else None

    # 8. Partition-isolated window generation
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
    if active_ts and active_ts in df.columns:
        from src.pipeline.windowing import verify_temporal_isolation

        isolation_audit = verify_temporal_isolation(
            train_df,
            val_df,
            test_df,
            timestamp_col=active_ts,
            duration_col=dur_col,
            duration_unit=dur_unit,
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
