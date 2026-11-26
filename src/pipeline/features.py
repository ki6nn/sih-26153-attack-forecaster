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


def extract_features(
    df: pd.DataFrame,
    dataset_name: str = "unsw_nb15",
    feature_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Extract, order, and cast the 20 features for model ingestion.

    Raises:
        ValueError: If required features are missing from df.
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

    extracted = df[target_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return extracted.astype(np.float32)


def map_stage_labels(
    labels: Sequence[Any] | pd.Series,
    dataset_name: str = "unsw_nb15",
) -> pd.Series:
    """Map raw categorical labels to standardized MITRE stages."""
    series = pd.Series(labels).astype(str).str.strip().str.lower()
    name = dataset_name.lower()
    mapping = CICIDS2017_LABEL_MAP if ("cic" in name or "ids" in name) else UNSW_NB15_LABEL_MAP

    mapped = series.map(mapping)
    # Default unmapped or unrecognized labels to 'Generic' or 'Normal' if benign
    unknown_mask = mapped.isna()
    if unknown_mask.any():
        for idx in series[unknown_mask].index:
            raw_val = series.loc[idx]
            if raw_val in {"0", "0.0", "benign", "normal", "nan", "", "none"}:
                mapped.loc[idx] = "Normal"
            else:
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
