"""Temporal windowing, explicit interval verification, and chronological data partitioning module."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.contracts import FORECAST_STEPS, WINDOW_SIZE


def validate_and_extract_durations(
    df: pd.DataFrame,
    duration_col: str | None,
    duration_unit: str = "s",
) -> pd.Series | None:
    """Validate and extract network flow active durations in seconds.

    Strict Requirements:
        - If duration_col is supplied, it must exist in df.
        - Non-numeric, NaN, infinite, or negative durations raise ValueError.
        - duration_unit must be strictly 's' (seconds) or 'us' (microseconds).
        - If duration_col is None, returns None (flow is treated as instantaneous event).

    Args:
        df: Input DataFrame.
        duration_col: Column name containing flow duration.
        duration_unit: 's' for seconds (UNSW-NB15) or 'us' for microseconds (CIC-IDS2017).

    Returns:
        pd.Series of duration in seconds (float64), or None if duration_col is None.

    Raises:
        ValueError: If duration_col is missing, unit is unsupported, or values are invalid.
    """
    if duration_col is None:
        return None

    if duration_col not in df.columns:
        raise ValueError(
            f"Specified duration column '{duration_col}' was not found in DataFrame. "
            f"Present columns: {list(df.columns[:10])}..."
        )

    unit = str(duration_unit).strip().lower()
    if unit not in {"s", "us"}:
        raise ValueError(
            f"Unsupported duration_unit '{duration_unit}'. Supported units are strictly "
            f"'s' (seconds) and 'us' (microseconds). Do not assume implicit units."
        )

    raw_durations = df[duration_col]
    if raw_durations.isna().any():
        nan_count = int(raw_durations.isna().sum())
        raise ValueError(
            f"Duration column '{duration_col}' contains {nan_count} NaN/null values. "
            f"Durations must be valid finite numbers."
        )

    try:
        numeric_durations = pd.to_numeric(raw_durations, errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Duration column '{duration_col}' contains non-numeric values: {exc}"
        ) from exc

    if np.isinf(numeric_durations).any():
        raise ValueError(f"Duration column '{duration_col}' contains infinite values.")

    if (numeric_durations < 0).any():
        neg_count = int((numeric_durations < 0).sum())
        raise ValueError(
            f"Duration column '{duration_col}' contains {neg_count} negative duration values. "
            f"Durations must be non-negative."
        )

    if unit == "us":
        # Convert microseconds to seconds
        return numeric_durations.astype(np.float64) / 1e6
    return numeric_durations.astype(np.float64)


def verify_temporal_isolation(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    timestamp_col: str = "datetime",
    duration_col: str | None = None,
    duration_unit: str = "s",
) -> dict[str, Any]:
    """Verify that training flows/targets strictly end before validation flows begin,

    and validation flows/targets strictly end before test flows begin.

    Args:
        train_df: Training partition.
        val_df: Validation partition.
        test_df: Test partition.
        timestamp_col: Column name containing flow start timestamps.
        duration_col: Optional column name containing flow durations.
        duration_unit: 's' for seconds, 'us' for microseconds.

    Returns:
        Audit dictionary with exact boundary timestamps and verified isolation gaps.

    Raises:
        ValueError: If temporal boundary violation, leakage, or overlap is detected.
    """
    for name, partition in (("Train", train_df), ("Validation", val_df), ("Test", test_df)):
        if len(partition) == 0:
            raise ValueError(f"Partition '{name}' is empty. Cannot verify temporal isolation.")
        if timestamp_col not in partition.columns:
            raise ValueError(f"Partition '{name}' is missing timestamp column '{timestamp_col}'.")
        if partition[timestamp_col].isna().any():
            raise ValueError(f"Partition '{name}' contains null or NaT timestamp values.")
        if not partition[timestamp_col].is_monotonic_increasing:
            raise ValueError(f"Partition '{name}' is not monotonically sorted by timestamp.")

    def _compute_flow_end_times(part: pd.DataFrame) -> pd.Series:
        start_ts = part[timestamp_col]
        dur_sec = validate_and_extract_durations(part, duration_col=duration_col, duration_unit=duration_unit)
        if dur_sec is not None:
            if pd.api.types.is_datetime64_any_dtype(start_ts):
                return start_ts + pd.to_timedelta(dur_sec, unit="s")
            return start_ts + dur_sec
        return start_ts

    train_end_time = _compute_flow_end_times(train_df).max()
    val_start_time = val_df[timestamp_col].min()
    val_end_time = _compute_flow_end_times(val_df).max()
    test_start_time = test_df[timestamp_col].min()

    # 1. Strict precedence: all training activity ends before validation activity starts
    if train_end_time >= val_start_time:
        raise ValueError(
            f"Temporal Leakage Detected! Training flow activity ends at {train_end_time}, "
            f"which overlaps or exceeds validation start at {val_start_time}."
        )

    # 2. Strict precedence: all validation activity ends before test activity starts
    if val_end_time >= test_start_time:
        raise ValueError(
            f"Temporal Leakage Detected! Validation flow activity ends at {val_end_time}, "
            f"which overlaps or exceeds test start at {test_start_time}."
        )

    # 3. Duplicate timestamp collision check across partition boundaries
    train_ts_set = set(train_df[timestamp_col].unique())
    val_ts_set = set(val_df[timestamp_col].unique())
    test_ts_set = set(test_df[timestamp_col].unique())

    train_val_overlap = train_ts_set.intersection(val_ts_set)
    if train_val_overlap:
        raise ValueError(
            f"Temporal Boundary Collision! Duplicate timestamps shared between Train and Validation: "
            f"{list(train_val_overlap)[:3]}"
        )

    val_test_overlap = val_ts_set.intersection(test_ts_set)
    if val_test_overlap:
        raise ValueError(
            f"Temporal Boundary Collision! Duplicate timestamps shared between Validation and Test: "
            f"{list(val_test_overlap)[:3]}"
        )

    return {
        "is_temporally_isolated": True,
        "train_range": (str(train_df[timestamp_col].min()), str(train_end_time)),
        "val_range": (str(val_start_time), str(val_end_time)),
        "test_range": (str(test_start_time), str(test_df[timestamp_col].max())),
        "train_val_gap_seconds": (
            (val_start_time - train_end_time).total_seconds()
            if hasattr(val_start_time - train_end_time, "total_seconds")
            else float(val_start_time - train_end_time)
        ),
        "val_test_gap_seconds": (
            (test_start_time - val_end_time).total_seconds()
            if hasattr(test_start_time - val_end_time, "total_seconds")
            else float(test_start_time - val_end_time)
        ),
    }


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    timestamp_col: str | None = None,
    duration_col: str | None = None,
    duration_unit: str = "s",
    min_gap_seconds: float = 0.0,
    buffer_size: int | None = None,
    allow_index_fallback: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition a DataFrame strictly by chronological time intervals and session end times.

    Guarantees:
        - Training flow activity (start + duration) strictly ends before validation begins.
        - Validation flow activity strictly ends before test begins.
        - Duplicate timestamps at partition boundaries are never split across sets.
        - Partitions are verified via verify_temporal_isolation() before returning.

    Args:
        df: Input DataFrame.
        train_ratio: Ratio for training set (default: 0.70).
        val_ratio: Ratio for validation set (default: 0.15).
        timestamp_col: Column name holding flow timestamps. If None, auto-detected.
        duration_col: Column name holding flow duration (e.g. 'dur' or 'flow_duration').
        duration_unit: 's' for seconds (UNSW-NB15) or 'us' for microseconds (CIC-IDS2017).
        min_gap_seconds: Optional additional elapsed-time separation between partitions.
        buffer_size: Optional minimum row buffer count.
        allow_index_fallback: If True, permits row-index splitting ONLY when timestamps
            are absent (e.g. synthetic demo smoke testing). Prohibited for research benchmarks.

    Returns:
        Tuple of (train_df, val_df, test_df).

    Raises:
        ValueError: If temporal ordering is violated, isolation fails, or partitions are empty.
    """
    total_len = len(df)
    if train_ratio <= 0.0 or val_ratio <= 0.0 or (train_ratio + val_ratio) >= 1.0:
        raise ValueError(
            f"Invalid split ratios: train={train_ratio}, val={val_ratio}. "
            f"Sum must be strictly less than 1.0."
        )

    # 1. Detect or validate timestamp column
    ts_col = timestamp_col
    if ts_col is None:
        for candidate in ["datetime", "timestamp", "stime", "flow_start_time", "time"]:
            if candidate in df.columns:
                ts_col = candidate
                break

    # Handle fallback if no timestamps are present
    if ts_col is None:
        if not allow_index_fallback:
            raise ValueError(
                "No valid timestamp column found in DataFrame. True temporal isolation "
                "requires flow timestamps (e.g. 'datetime', 'timestamp', 'stime'). "
                "Set allow_index_fallback=True only for non-research offline smoke testing."
            )
        # Mechanical index fallback for synthetic demo data
        row_buffer = buffer_size if buffer_size is not None else (WINDOW_SIZE + FORECAST_STEPS - 1)
        train_end = int(total_len * train_ratio)
        val_start = train_end + row_buffer
        val_end = int(total_len * (train_ratio + val_ratio))
        test_start = val_end + row_buffer
        if val_start >= val_end or test_start >= total_len:
            raise ValueError("Insufficient rows for index fallback splitting.")
        return (
            df.iloc[:train_end].copy().reset_index(drop=True),
            df.iloc[val_start:val_end].copy().reset_index(drop=True),
            df.iloc[test_start:].copy().reset_index(drop=True),
        )

    # 2. Strict timestamp validations
    if df[ts_col].isna().any():
        raise ValueError(f"Timestamp column '{ts_col}' contains null/NaT values. Data must be sanitized first.")
    if not df[ts_col].is_monotonic_increasing:
        raise ValueError(f"DataFrame is not sorted chronologically by '{ts_col}'. Sort before splitting.")

    # 3. Compute flow completion times accounting for duration
    start_ts = df[ts_col]
    dur_sec = validate_and_extract_durations(df, duration_col=duration_col, duration_unit=duration_unit)
    if dur_sec is not None:
        if pd.api.types.is_datetime64_any_dtype(start_ts):
            end_ts = start_ts + pd.to_timedelta(dur_sec, unit="s")
        else:
            end_ts = start_ts + dur_sec
    else:
        end_ts = start_ts

    # 4. Determine initial split targets
    min_rows_per_split = WINDOW_SIZE + FORECAST_STEPS
    min_total_required = min_rows_per_split * 3
    if total_len < min_total_required:
        raise ValueError(
            f"Dataset length ({total_len}) is insufficient. Minimum {min_total_required} rows required."
        )

    # Find train boundary: keep duplicate timestamps together
    target_train_idx = max(min_rows_per_split, int(total_len * train_ratio))
    cut_ts = start_ts.iloc[target_train_idx - 1]
    # Advance to end of the duplicate timestamp group
    k1 = target_train_idx - 1
    while k1 + 1 < total_len and start_ts.iloc[k1 + 1] == cut_ts:
        k1 += 1

    # If advancing past cut_ts leaves insufficient rows for val and test, try ending before cut_ts
    if (total_len - (k1 + 1)) < 2 * min_rows_per_split:
        first_cut_idx = int(start_ts.searchsorted(cut_ts, side="left"))
        if first_cut_idx >= min_rows_per_split:
            k1 = first_cut_idx - 1
        else:
            raise ValueError(
                f"Duplicate timestamp cluster at {cut_ts} is too large to partition. "
                f"Cannot allocate minimum {min_rows_per_split} rows to each partition."
            )

    train_slice = df.iloc[: k1 + 1]
    # Max completion time in training
    train_max_completion = end_ts.iloc[: k1 + 1].max()
    if min_gap_seconds > 0:
        if pd.api.types.is_datetime64_any_dtype(train_max_completion):
            train_max_completion = train_max_completion + pd.to_timedelta(min_gap_seconds, unit="s")
        else:
            train_max_completion = train_max_completion + min_gap_seconds

    # Validation must start at the first record whose start time is strictly AFTER train_max_completion
    val_start_idx = k1 + 1
    while val_start_idx < total_len and start_ts.iloc[val_start_idx] <= train_max_completion:
        val_start_idx += 1

    # Apply row buffer if explicitly requested
    if buffer_size is not None and (val_start_idx - (k1 + 1)) < buffer_size:
        val_start_idx = k1 + 1 + buffer_size
        while val_start_idx < total_len and start_ts.iloc[val_start_idx] <= train_max_completion:
            val_start_idx += 1

    # Find validation end boundary
    target_val_idx = max(val_start_idx + min_rows_per_split, int(total_len * (train_ratio + val_ratio)))
    if target_val_idx >= total_len - min_rows_per_split:
        target_val_idx = val_start_idx + min_rows_per_split

    if target_val_idx >= total_len:
        raise ValueError(
            f"Cannot allocate isolated validation partition. Validation target index ({target_val_idx}) "
            f"exceeds dataset size ({total_len})."
        )

    val_cut_ts = start_ts.iloc[target_val_idx - 1]
    k2 = target_val_idx - 1
    while k2 + 1 < total_len and start_ts.iloc[k2 + 1] == val_cut_ts:
        k2 += 1

    # If advancing past val_cut_ts leaves insufficient rows for test, try ending before val_cut_ts
    if (total_len - (k2 + 1)) < min_rows_per_split:
        first_val_cut_idx = int(start_ts.searchsorted(val_cut_ts, side="left"))
        if (first_val_cut_idx - val_start_idx) >= min_rows_per_split:
            k2 = first_val_cut_idx - 1
        else:
            raise ValueError(
                f"Duplicate timestamp cluster at {val_cut_ts} is too large to partition. "
                f"Leaves insufficient records for test partition."
            )

    val_slice = df.iloc[val_start_idx : k2 + 1]
    val_max_completion = end_ts.iloc[val_start_idx : k2 + 1].max()
    if min_gap_seconds > 0:
        if pd.api.types.is_datetime64_any_dtype(val_max_completion):
            val_max_completion = val_max_completion + pd.to_timedelta(min_gap_seconds, unit="s")
        else:
            val_max_completion = val_max_completion + min_gap_seconds

    # Test must start at the first record whose start time is strictly AFTER val_max_completion
    test_start_idx = k2 + 1
    while test_start_idx < total_len and start_ts.iloc[test_start_idx] <= val_max_completion:
        test_start_idx += 1

    if buffer_size is not None and (test_start_idx - (k2 + 1)) < buffer_size:
        test_start_idx = k2 + 1 + buffer_size
        while test_start_idx < total_len and start_ts.iloc[test_start_idx] <= val_max_completion:
            test_start_idx += 1

    test_slice = df.iloc[test_start_idx :]

    # 5. Enforce minimum partition lengths
    if len(train_slice) < min_rows_per_split:
        raise ValueError(f"Training partition has {len(train_slice)} rows; minimum {min_rows_per_split} required.")
    if len(val_slice) < min_rows_per_split:
        raise ValueError(f"Validation partition has {len(val_slice)} rows; minimum {min_rows_per_split} required.")
    if len(test_slice) < min_rows_per_split:
        raise ValueError(f"Test partition has {len(test_slice)} rows; minimum {min_rows_per_split} required.")

    # 6. Active production temporal verification
    train_df = train_slice.copy().reset_index(drop=True)
    val_df = val_slice.copy().reset_index(drop=True)
    test_df = test_slice.copy().reset_index(drop=True)

    verify_temporal_isolation(
        train_df,
        val_df,
        test_df,
        timestamp_col=ts_col,
        duration_col=duration_col,
        duration_unit=duration_unit,
    )

    return train_df, val_df, test_df


def create_network_windows(
    X: np.ndarray | pd.DataFrame,
    y_attack: np.ndarray | pd.Series | Sequence[int],
    y_stage: np.ndarray | pd.Series | Sequence[int] | None = None,
    timestamps: np.ndarray | pd.Series | Sequence[Any] | None = None,
    window_size: int = WINDOW_SIZE,
    forecast_steps: int = FORECAST_STEPS,
    return_intervals: bool = False,
    return_future_states: bool = False,
) -> tuple[Any, ...]:
    """Construct sliding feature windows and multi-step future targets strictly within a partition.

    Explicit interval definition:
        For window index i:
            - Lookback Input Interval:  [i, i + window_size - 1]
            - Forecast Target Interval: [i + window_size, i + window_size + forecast_steps - 1]
            - Prediction Issuance Time: step (i + window_size - 1)

    Args:
        X: Feature matrix of shape (N, D).
        y_attack: Binary attack labels of shape (N,). 1 if attack, 0 if benign.
        y_stage: Optional integer category labels of shape (N,).
        timestamps: Optional timestamps of shape (N,) for tracking temporal intervals.
        window_size: Past lookback timesteps (W=10).
        forecast_steps: Future prediction horizon (H=4).
        return_intervals: If True, returns an interval audit DataFrame.
        return_future_states: If True, returns future feature states X_future of shape (M, H, D)
            and aligned multi-horizon category targets of shape (M, H).

    Returns:
        If return_future_states=True:
            (X_windows, X_future, y_future_attack, y_future_stage [, interval_df])
        If return_future_states=False (legacy parity):
            (X_windows, y_future_attack, y_next_stage [, interval_df])

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

    X_future = np.stack([X_arr[i + window_size : i + total_horizon] for i in range(valid_windows)])
    y_future_stage = np.stack([y_stg_arr[i + window_size : i + total_horizon] for i in range(valid_windows)])

    ts_arr = np.asarray(timestamps) if timestamps is not None else None
    interval_df = None
    if return_intervals:
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

    if return_future_states:
        if return_intervals:
            return (
                X_windows.astype(np.float32),
                X_future.astype(np.float32),
                y_future_attack.astype(np.float32),
                y_future_stage.astype(np.int64),
                interval_df,
            )
        return (
            X_windows.astype(np.float32),
            X_future.astype(np.float32),
            y_future_attack.astype(np.float32),
            y_future_stage.astype(np.int64),
        )

    if return_intervals:
        return (
            X_windows.astype(np.float32),
            y_future_attack.astype(np.float32),
            y_next_stage.astype(np.int64),
            interval_df,
        )
    return (
        X_windows.astype(np.float32),
        y_future_attack.astype(np.float32),
        y_next_stage.astype(np.int64),
    )
