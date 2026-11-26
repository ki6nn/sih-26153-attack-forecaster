"""Temporal windowing, explicit interval verification, and chronological data partitioning module."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.contracts import FORECAST_STEPS, WINDOW_SIZE


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    timestamp_col: str | None = None,
    buffer_size: int = WINDOW_SIZE + FORECAST_STEPS - 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition a DataFrame strictly by chronological order with safety boundary exclusions.

    Args:
        df: Input DataFrame (must be chronologically ordered).
        train_ratio: Fraction of records allocated to training (default: 0.70).
        val_ratio: Fraction of records allocated to validation (default: 0.15).
        timestamp_col: Optional column name holding timestamps. If provided, ensures
            identical timestamps are never split across partition boundaries.
        buffer_size: Minimum number of records to discard at partition boundaries
            to prevent sliding-window sequence lookahead or state contamination.

    Returns:
        Tuple of (train_df, val_df, test_df).

    Raises:
        ValueError: If ratios are invalid or dataset has insufficient records.
    """
    total_len = len(df)
    if train_ratio <= 0.0 or val_ratio <= 0.0 or (train_ratio + val_ratio) >= 1.0:
        raise ValueError(
            f"Invalid split ratios: train={train_ratio}, val={val_ratio}. "
            f"Sum must be strictly less than 1.0."
        )

    min_required = 3 * (WINDOW_SIZE + FORECAST_STEPS) + 2 * buffer_size
    if total_len < min_required:
        raise ValueError(
            f"Dataset length ({total_len}) is too short for safe chronological splitting. "
            f"Minimum required rows: {min_required}."
        )

    train_end = int(total_len * train_ratio)
    val_start = train_end + buffer_size
    val_end = int(total_len * (train_ratio + val_ratio))
    test_start = val_end + buffer_size

    # Guard against splitting across duplicate timestamps at boundary
    if timestamp_col and timestamp_col in df.columns:
        ts_series = df[timestamp_col]
        # If train_end falls in a group of duplicate timestamps, push val_start past that timestamp
        train_ts = ts_series.iloc[train_end - 1]
        while val_start < total_len and ts_series.iloc[val_start] == train_ts:
            val_start += 1

        val_ts = ts_series.iloc[val_end - 1] if val_end - 1 < total_len else None
        if val_ts is not None:
            while test_start < total_len and ts_series.iloc[test_start] == val_ts:
                test_start += 1

    if val_start >= val_end or test_start >= total_len:
        raise ValueError(
            f"Insufficient records to accommodate buffer_size={buffer_size} between partitions. "
            f"Rows: {total_len}, train_end: {train_end}, val_range: [{val_start}, {val_end}], "
            f"test_start: {test_start}"
        )

    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[val_start:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[test_start:].copy().reset_index(drop=True)

    return train_df, val_df, test_df


def create_network_windows(
    X: np.ndarray | pd.DataFrame,
    y_attack: np.ndarray | pd.Series | Sequence[int],
    y_stage: np.ndarray | pd.Series | Sequence[int] | None = None,
    timestamps: np.ndarray | pd.Series | Sequence[Any] | None = None,
    window_size: int = WINDOW_SIZE,
    forecast_steps: int = FORECAST_STEPS,
    return_intervals: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Construct sliding feature windows and multi-step future targets strictly within a partition.

    Explicit interval definition:
        For window index i:
            - Lookback Input Interval:  [i, i + window_size - 1]
            - Forecast Target Interval: [i + window_size, i + window_size + forecast_steps - 1]
            - Prediction Issuance Time: step (i + window_size - 1)

    Args:
        X: Feature matrix of shape (N, D).
        y_attack: Binary attack labels of shape (N,). 1 if attack, 0 if benign.
        y_stage: Optional integer MITRE stage labels of shape (N,).
        timestamps: Optional timestamps of shape (N,) for tracking temporal intervals.
        window_size: Past lookback timesteps (W=10).
        forecast_steps: Future prediction horizon (H=4).
        return_intervals: If True, returns an interval audit DataFrame.

    Returns:
        Tuple of:
            - X_windows: (M, W, D) float32 sliding feature history.
            - y_future_attack: (M, H) float32 multi-step future binary labels.
            - y_next_stage: (M,) int64 immediate next-step stage label.
            - (Optional) interval_df: Metadata DataFrame with exact start/end indices and timestamps.

    Raises:
        ValueError: If input sample count is less than (window_size + forecast_steps).
    """
    X_arr = X.to_numpy(dtype=np.float32) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=np.float32)
    y_att_arr = np.asarray(y_attack, dtype=np.float32).reshape(-1)

    if X_arr.ndim != 2:
        raise ValueError(f"Features array X must be 2-dimensional (N, D), got shape {X_arr.shape}")
    if X_arr.shape[0] != y_att_arr.shape[0]:
        raise ValueError(
            f"Sample count mismatch: X has {X_arr.shape[0]} rows, y_attack has {y_att_arr.shape[0]}"
        )

    n_samples = X_arr.shape[0]
    total_horizon = window_size + forecast_steps
    valid_windows = n_samples - total_horizon + 1

    if valid_windows <= 0:
        raise ValueError(
            f"Input length {n_samples} is too short. "
            f"Minimum {total_horizon} steps required (W={window_size} + H={forecast_steps})."
        )

    if y_stage is not None:
        y_stg_arr = np.asarray(y_stage, dtype=np.int64).reshape(-1)
        if y_stg_arr.shape[0] != n_samples:
            raise ValueError("y_stage length must match X sample count")
    else:
        y_stg_arr = (y_att_arr > 0).astype(np.int64)

    X_windows = np.stack([X_arr[i : i + window_size] for i in range(valid_windows)])
    y_future_attack = np.stack(
        [y_att_arr[i + window_size : i + total_horizon] for i in range(valid_windows)]
    )
    y_next_stage = y_stg_arr[window_size : window_size + valid_windows]

    if not return_intervals:
        return X_windows.astype(np.float32), y_future_attack.astype(np.float32), y_next_stage.astype(np.int64)

    ts_arr = np.asarray(timestamps) if timestamps is not None else None
    records = []
    for i in range(valid_windows):
        input_start = i
        input_end = i + window_size - 1
        target_start = i + window_size
        target_end = i + total_horizon - 1
        rec: dict[str, Any] = {
            "window_id": i,
            "input_start_idx": input_start,
            "input_end_idx": input_end,
            "target_start_idx": target_start,
            "target_end_idx": target_end,
        }
        if ts_arr is not None:
            rec["input_start_time"] = ts_arr[input_start]
            rec["input_end_time"] = ts_arr[input_end]
            rec["target_start_time"] = ts_arr[target_start]
            rec["target_end_time"] = ts_arr[target_end]
        records.append(rec)

    interval_df = pd.DataFrame.from_records(records)
    return (
        X_windows.astype(np.float32),
        y_future_attack.astype(np.float32),
        y_next_stage.astype(np.int64),
        interval_df,
    )


def verify_temporal_isolation(
    train_targets_end: Any,
    val_inputs_start: Any,
    val_targets_end: Any,
    test_inputs_start: Any,
) -> bool:
    """Explicitly verify that training targets never overlap validation inputs,

    and validation targets never overlap test inputs.

    Raises:
        ValueError: If temporal boundary violation is detected.
    """
    if train_targets_end >= val_inputs_start:
        raise ValueError(
            f"Temporal Leakage Detected! Training target end ({train_targets_end}) "
            f"must strictly precede validation input start ({val_inputs_start})."
        )
    if val_targets_end >= test_inputs_start:
        raise ValueError(
            f"Temporal Leakage Detected! Validation target end ({val_targets_end}) "
            f"must strictly precede test input start ({test_inputs_start})."
        )
    return True
