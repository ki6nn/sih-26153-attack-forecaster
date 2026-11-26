"""Data validation, anomaly detection, and quality assessment module."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


def validate_required_columns(df: pd.DataFrame, required_columns: Sequence[str]) -> None:
    """Verify that all specified columns are present in the DataFrame.

    Raises:
        ValueError: If any required columns are missing.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns ({len(missing)}): {', '.join(missing)}. "
            f"Present columns: {list(df.columns[:10])}..."
        )


def detect_missing_and_infinite(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> dict[str, dict[str, int | float]]:
    """Detect NaN, null, and infinite values across specified or all numeric columns.

    Returns:
        Dictionary mapping column name to counts and percentages of missing and infinite values.
    """
    target_cols = list(columns) if columns else list(df.select_dtypes(include=[np.number]).columns)
    report = {}
    total_rows = len(df)

    for col in target_cols:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        nan_count = int(series.isna().sum())
        inf_count = int(np.isinf(series).sum()) if hasattr(np, "isinf") else 0
        report[col] = {
            "nan_count": nan_count,
            "nan_pct": float(nan_count / total_rows * 100.0) if total_rows > 0 else 0.0,
            "inf_count": inf_count,
            "inf_pct": float(inf_count / total_rows * 100.0) if total_rows > 0 else 0.0,
        }
    return report


def clean_missing_and_infinite(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    strategy: str = "median",
    fill_values: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Sanitize data by replacing infinities and imputing missing values.

    Args:
        df: Input DataFrame.
        columns: Target numeric columns to sanitize.
        strategy: 'median', 'mean', or 'zero' (used if fill_values is None).
        fill_values: Precomputed imputation statistics (from training split).

    Returns:
        Tuple of (cleaned_df, computed_fill_values).
    """
    cleaned = df.copy()
    target_cols = list(columns) if columns else list(cleaned.select_dtypes(include=[np.number]).columns)
    computed_fills: dict[str, float] = {}

    for col in target_cols:
        if col not in cleaned.columns:
            continue
        series = pd.to_numeric(cleaned[col], errors="coerce")
        series = series.replace([np.inf, -np.inf], np.nan)

        if fill_values and col in fill_values:
            fill = fill_values[col]
        else:
            if strategy == "median":
                fill = float(series.median()) if not series.dropna().empty else 0.0
            elif strategy == "mean":
                fill = float(series.mean()) if not series.dropna().empty else 0.0
            else:
                fill = 0.0
            computed_fills[col] = fill

        cleaned[col] = series.fillna(fill).astype(np.float32)

    return cleaned, (fill_values or computed_fills)


def validate_labels(
    df: pd.DataFrame,
    label_col: str = "attack_cat",
    valid_labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Audit the label column for nulls, cardinality, and unexpected classes."""
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame")

    series = df[label_col].fillna("Missing").astype(str)
    counts = series.value_counts().to_dict()
    missing_count = int((series == "Missing").sum())
    unique_labels = list(counts.keys())

    unexpected = []
    if valid_labels:
        valid_set = {str(lbl).lower() for lbl in valid_labels}
        unexpected = [lbl for lbl in unique_labels if lbl.lower() not in valid_set and lbl != "Missing"]

    return {
        "label_column": label_col,
        "total_records": len(df),
        "unique_classes": len(unique_labels),
        "missing_labels": missing_count,
        "class_distribution": counts,
        "unexpected_classes": unexpected,
        "is_valid": len(unexpected) == 0 and missing_count == 0,
    }


def detect_duplicate_records(
    df: pd.DataFrame,
    subset: Sequence[str] | None = None,
) -> tuple[int, pd.Series]:
    """Detect exact duplicate flow records.

    Returns:
        Tuple of (duplicate_count, boolean_mask_of_duplicates).
    """
    dup_mask = df.duplicated(subset=subset, keep="first")
    return int(dup_mask.sum()), dup_mask


def generate_data_quality_report(
    df: pd.DataFrame,
    dataset_name: str | None = None,
    numeric_columns: Sequence[str] | None = None,
    label_col: str | None = None,
) -> dict[str, Any]:
    """Generate a comprehensive health and data quality report for network flow logs."""
    total_rows = len(df)
    missing_inf = detect_missing_and_infinite(df, columns=numeric_columns)
    dup_count, _ = detect_duplicate_records(df, subset=numeric_columns)

    temporal_info: dict[str, Any] = {}
    if "datetime" in df.columns and not df["datetime"].isna().all():
        valid_dates = df["datetime"].dropna()
        temporal_info = {
            "start": str(valid_dates.min()),
            "end": str(valid_dates.max()),
            "duration_seconds": float((valid_dates.max() - valid_dates.min()).total_seconds()),
            "is_monotonic_increasing": bool(df["datetime"].is_monotonic_increasing),
        }

    label_info: dict[str, Any] = {}
    if label_col and label_col in df.columns:
        label_info = validate_labels(df, label_col=label_col)

    return {
        "dataset_name": dataset_name or df.get("_source_dataset", pd.Series(["unknown"])).iloc[0]
        if "_source_dataset" in df.columns and len(df) > 0
        else "unknown",
        "total_rows": total_rows,
        "total_columns": len(df.columns),
        "duplicate_rows": dup_count,
        "duplicate_pct": float(dup_count / total_rows * 100.0) if total_rows > 0 else 0.0,
        "columns_with_missing_or_inf": {
            col: stats
            for col, stats in missing_inf.items()
            if stats["nan_count"] > 0 or stats["inf_count"] > 0
        },
        "temporal_coverage": temporal_info,
        "label_health": label_info,
    }
