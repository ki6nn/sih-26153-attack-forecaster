"""Feature definition, extraction, categorical encoding, and label mapping."""

from __future__ import annotations

from typing import Any, Final, Sequence

import numpy as np
import pandas as pd

from src.contracts import MITRE_STAGE_MAP, NUM_FEATURES

# Documented 20 flow features for CIC-IDS2017
CICIDS2017_FEATURES: Final[list[str]] = [
    "flow_duration",
    "total_fwd_packets",
    "total_backward_packets",
    "total_length_of_fwd_packets",
    "total_length_of_bwd_packets",
    "fwd_packet_length_max",
    "fwd_packet_length_min",
    "fwd_packet_length_mean",
    "bwd_packet_length_max",
    "bwd_packet_length_min",
    "bwd_packet_length_mean",
    "flow_bytes_s",
    "flow_packets_s",
    "flow_iat_mean",
    "flow_iat_std",
    "fwd_iat_total",
    "bwd_iat_total",
    "fin_flag_count",
    "syn_flag_count",
    "init_win_bytes_forward",
]

# Documented 20 flow features for UNSW-NB15
UNSW_NB15_FEATURES: Final[list[str]] = [
    "dur",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sttl",
    "dttl",
    "sload",
    "dload",
    "sloss",
    "dloss",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "swin",
    "dwin",
    "ct_state_ttl",
    "ct_srv_src",
]

# Mapping rules from raw dataset labels to standardized MITRE ATT&CK stages
CICIDS2017_LABEL_MAP: Final[dict[str, str]] = {
    "benign": "Normal",
    "portscan": "Reconnaissance",
    "bot": "Analysis",
    "web attack – brute force": "Exploits",
    "web attack – xss": "Exploits",
    "web attack – sql injection": "Exploits",
    "web attack brute force": "Exploits",
    "web attack xss": "Exploits",
    "web attack sql injection": "Exploits",
    "dos slowloris": "DoS",
    "dos slowhttptest": "DoS",
    "dos hulk": "DoS",
    "dos goldeneye": "DoS",
    "heartbleed": "DoS",
    "ddos": "DoS",
    "ftp-patator": "Generic",
    "ssh-patator": "Generic",
    "infiltration": "Backdoor",
}

UNSW_NB15_LABEL_MAP: Final[dict[str, str]] = {
    "normal": "Normal",
    "reconnaissance": "Reconnaissance",
    "analysis": "Analysis",
    "fuzzers": "Fuzzers",
    "exploits": "Exploits",
    "dos": "DoS",
    "generic": "Generic",
    "shellcode": "Shellcode",
    "backdoors": "Backdoor",
    "backdoor": "Backdoor",
    "worms": "Worms",
}


def get_feature_columns(dataset_name: str = "unsw_nb15") -> list[str]:
    """Return the documented 20 feature names for the dataset."""
    name = dataset_name.lower()
    if "cic" in name or "ids" in name:
        return list(CICIDS2017_FEATURES)
    return list(UNSW_NB15_FEATURES)


def detect_feature_anomalies(
    df: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
) -> dict[str, dict[str, int]]:
    """Detect missing, malformed, NaN, positive infinity, and negative infinity values.

    Args:
        df: Input DataFrame.
        feature_columns: Feature columns to inspect.

    Returns:
        Dictionary mapping column name to anomaly counts:
        {'nan_count': int, 'posinf_count': int, 'neginf_count': int, 'non_numeric_count': int}
    """
    cols = list(feature_columns) if feature_columns else list(df.columns)
    anomalies: dict[str, dict[str, int]] = {}

    for col in cols:
        if col not in df.columns:
            continue
        series = df[col]
        # Detect non-numeric
        coerced = pd.to_numeric(series, errors="coerce")
        non_numeric_count = int(series.notna().sum() - coerced.notna().sum())
        nan_count = int(coerced.isna().sum()) - non_numeric_count
        posinf_count = int(np.isposinf(coerced.to_numpy(dtype=np.float64, na_value=0.0)).sum())
        neginf_count = int(np.isneginf(coerced.to_numpy(dtype=np.float64, na_value=0.0)).sum())

        total_anomalies = non_numeric_count + nan_count + posinf_count + neginf_count
        if total_anomalies > 0:
            anomalies[col] = {
                "nan_count": nan_count,
                "posinf_count": posinf_count,
                "neginf_count": neginf_count,
                "non_numeric_count": non_numeric_count,
                "total_invalid": total_anomalies,
            }

    return anomalies


def extract_features(
    df: pd.DataFrame,
    dataset_name: str = "unsw_nb15",
    feature_columns: Sequence[str] | None = None,
    impute_values: dict[str, float] | None = None,
    raise_on_invalid: bool = False,
) -> pd.DataFrame:
    """Extract, order, and cast the 20 features for model ingestion.

    Enforces explicit numeric validation and training-only imputation:
        - If invalid values (NaN, +Inf, -Inf, malformed) are detected:
            * If impute_values is supplied: fills with the exact pre-fitted training values.
            * If raise_on_invalid=True and impute_values is None: raises ValueError detailing
              affected columns and counts.
            * Otherwise: falls back to 0.0 only for non-research offline smoke testing.
        - Legitimate zero-valued network features are preserved untouched.

    Raises:
        ValueError: If required features are missing or invalid values are present without imputation.
    """
    target_cols = list(feature_columns) if feature_columns else get_feature_columns(dataset_name)
    if len(target_cols) != NUM_FEATURES:
        raise ValueError(
            f"Expected {NUM_FEATURES} features, but target list has {len(target_cols)}"
        )

    missing = [c for c in target_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame is missing required {dataset_name} features: {', '.join(missing)}"
        )

    extracted = pd.DataFrame(index=df.index)
    anomalies = detect_feature_anomalies(df, target_cols)

    if anomalies and raise_on_invalid and impute_values is None:
        summary = {col: counts["total_invalid"] for col, counts in anomalies.items()}
        raise ValueError(
            f"Invalid numeric feature values detected in {len(anomalies)} columns without an "
            f"approved imputation policy: {summary}. In research mode, use training-fitted "
            f"imputation via NetworkDataPreprocessor."
        )

    for col in target_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        # Replace non-finite values (NaN, +Inf, -Inf)
        non_finite_mask = series.isna() | np.isinf(series.to_numpy(dtype=np.float64, na_value=0.0))
        if non_finite_mask.any():
            fill_val = 0.0
            if impute_values is not None and col in impute_values:
                fill_val = float(impute_values[col])
            series = series.where(~non_finite_mask, fill_val)
        extracted[col] = series.astype(np.float32)

    return extracted


def map_stage_labels(
    labels: Sequence[Any] | pd.Series,
    dataset_name: str = "unsw_nb15",
    strict: bool = False,
    allow_unknown: bool = True,
) -> pd.Series:
    """Map raw categorical labels to standardized MITRE stages.

    Policy:
        - Null, empty string, NaN, or 'none' labels raise ValueError. Target labels cannot
          silently become benign (Normal).
        - Unknown attack categories map to 'Generic' (attack) or raise ValueError if strict=True.
          Unknown categories are NEVER mapped to 'Normal'.
        - MITRE stage mapping is a high-level heuristic tactical categorization; it does NOT
          claim verified MITRE ATT&CK enterprise technique attributions.

    Raises:
        ValueError: If null/empty labels are encountered, or unknown labels in strict mode.
    """
    raw_s = pd.Series(labels)
    # Check for null, NaN, or empty values
    null_mask = raw_s.isna() | raw_s.astype(str).str.strip().str.lower().isin(["", "nan", "none", "null"])
    if null_mask.any():
        null_count = int(null_mask.sum())
        raise ValueError(
            f"Target labels contain {null_count} null/empty/NaN values. Target labels must be "
            f"fully observed and cannot be silently imputed as benign (Normal)."
        )

    series = raw_s.astype(str).str.strip().str.lower()
    name = dataset_name.lower()
    mapping = CICIDS2017_LABEL_MAP if ("cic" in name or "ids" in name) else UNSW_NB15_LABEL_MAP

    mapped = series.map(mapping)
    unknown_mask = mapped.isna()
    if unknown_mask.any():
        unknown_labels = sorted(series[unknown_mask].unique().tolist())
        if strict and not allow_unknown:
            raise ValueError(
                f"Unknown attack category labels detected for {dataset_name}: {unknown_labels}. "
                f"Strict mode requires all labels to exist in documented dataset mapping."
            )
        # Default unknown non-empty attack labels to 'Generic' (an attack stage), NEVER to 'Normal'
        for idx in series[unknown_mask].index:
            mapped.loc[idx] = "Generic"

    return mapped.astype(str)


def encode_categorical_features(
    df: pd.DataFrame,
    categorical_columns: Sequence[str],
    encoders: dict[str, dict[str, int]] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Deterministically encode categorical columns using integer mapping without target leakage.

    Args:
        df: Input DataFrame.
        categorical_columns: Columns to encode (e.g. ['proto', 'service', 'state']).
        encoders: Pre-fitted category mapping (from training partition). If None, fits from df.

    Returns:
        Tuple of (encoded_df, fitted_encoders).
    """
    result = df.copy()
    fitted_encoders: dict[str, dict[str, int]] = {} if encoders is None else encoders

    for col in categorical_columns:
        if col not in result.columns:
            continue

        str_series = result[col].fillna("UNKNOWN").astype(str).str.lower()
        if encoders is None or col not in encoders:
            # Fit categories deterministically sorted
            unique_vals = sorted(str_series.unique().tolist())
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            # Reserve 0 for unknown if not present
            if "unknown" not in mapping:
                mapping = {"unknown": 0, **{v: i + 1 for i, v in enumerate(unique_vals)}}
            fitted_encoders[col] = mapping
        else:
            mapping = encoders[col]

        unknown_code = mapping.get("unknown", 0)
        result[col] = str_series.map(mapping).fillna(unknown_code).astype(np.int64)

    return result, fitted_encoders
