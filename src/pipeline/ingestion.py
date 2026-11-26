"""Dataset ingestion module for loading and standardizing raw network traffic CSVs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Generator, Sequence

import pandas as pd


def normalize_column_names(columns: Sequence[str] | pd.DataFrame) -> list[str] | pd.DataFrame:
    """Standardize column names by stripping whitespace, lowercasing, and replacing special characters.

    Examples:
        ' Destination Port' -> 'destination_port'
        'Flow Bytes/s'      -> 'flow_bytes_s'
        'Attack_Cat'        -> 'attack_cat'
        'Fwd IAT Total'     -> 'fwd_iat_total'
    """
    if isinstance(columns, pd.DataFrame):
        df = columns.copy()
        df.columns = normalize_column_names(df.columns)
        return df

    normalized = []
    for col in columns:
        clean = str(col).strip().lower()
        clean = re.sub(r"[ /\\-]+", "_", clean)
        clean = re.sub(r"[^a-z0-9_]", "", clean)
        clean = re.sub(r"_+", "_", clean).strip("_")
        normalized.append(clean)
    return normalized


def parse_timestamps(
    df: pd.DataFrame,
    timestamp_col: str | None = None,
    sort_chronological: bool = True,
    strict: bool = False,
) -> pd.DataFrame:
    """Identify and parse timestamp column into standardized UTC datetimes while preserving original raw values.

    Distinguishes:
        - ISO datetime strings (e.g. CIC-IDS2017 '07/07/2017 08:30:00' or ISO-8601)
        - Numeric epoch timestamps in seconds (UNSW-NB15 stime/ltime ~ 1.42e9)
        - Numeric epoch timestamps in milliseconds (~ 1.42e12)
        - Relative seconds from start of capture (< 1e8)

    Preserves the original source column untouched and adds 'datetime' column.
    Uses stable sorting so identical timestamp groups preserve original arrival order.

    Raises:
        ValueError: If strict=True and timestamp parsing produces NaT/null values.
    """
    result = df.copy()
    candidate_cols = (
        [timestamp_col]
        if timestamp_col
        else ["timestamp", "stime", "ltime", "flow_start_time", "time", "date"]
    )

    detected_col = None
    for col in candidate_cols:
        if col in result.columns:
            detected_col = col
            break

    if detected_col is None:
        return result

    raw_series = result[detected_col]
    non_null_samples = raw_series.dropna()
    if len(non_null_samples) == 0:
        result["datetime"] = pd.Series([pd.NaT] * len(result), dtype="datetime64[ns, UTC]")
        if strict:
            raise ValueError(f"Timestamp column '{detected_col}' contains only null values.")
        return result

    is_numeric = False
    try:
        pd.to_numeric(non_null_samples.iloc[:20])
        is_numeric = True
    except (ValueError, TypeError):
        is_numeric = False

    if is_numeric:
        raw_numeric = pd.to_numeric(raw_series, errors="coerce")
        median_val = float(raw_numeric.dropna().median()) if len(raw_numeric.dropna()) > 0 else 0.0
        # Distinguish millisecond epochs from second epochs
        if median_val > 1e11:
            unit = "ms"
        elif median_val > 1e8:
            unit = "s"
        else:
            # Relative flow seconds starting from zero: anchor to arbitrary base epoch
            unit = "s"
        result["datetime"] = pd.to_datetime(raw_numeric, unit=unit, utc=True, errors="coerce")
    else:
        # String timestamps (e.g. CIC-IDS2017 '07/07/2017 09:30:00')
        result["datetime"] = pd.to_datetime(
            raw_series, utc=True, errors="coerce", format="mixed"
        )

    nat_count = int(result["datetime"].isna().sum())
    if strict and nat_count > int(raw_series.isna().sum()):
        raise ValueError(
            f"Timestamp parsing failed for {nat_count - int(raw_series.isna().sum())} records "
            f"in column '{detected_col}'."
        )

    if sort_chronological and not result["datetime"].isna().all():
        # Stable sort preserves identical timestamp groupings in original relative order
        result = result.sort_values(by="datetime", ascending=True, kind="stable").reset_index(drop=True)

    return result


def record_source_provenance(
    df: pd.DataFrame,
    dataset_name: str,
    filepath: str | Path,
) -> pd.DataFrame:
    """Attach dataset origin and file provenance metadata to the dataframe."""
    result = df.copy()
    result["_source_dataset"] = str(dataset_name)
    result["_source_file"] = Path(filepath).name
    return result


def load_csv_chunks(
    filepath: str | Path,
    chunksize: int = 50000,
    nrows: int | None = None,
    usecols: list[str] | None = None,
) -> Generator[pd.DataFrame, None, None]:
    """Yield standardized chunks from a CSV file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found at: {path}")

    rows_read = 0
    for chunk in pd.read_csv(path, chunksize=chunksize, usecols=usecols, low_memory=False):
        chunk = normalize_column_names(chunk)
        if nrows is not None:
            remaining = nrows - rows_read
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining]
        rows_read += len(chunk)
        yield chunk


def load_dataset(
    filepath: str | Path,
    dataset_name: str = "auto",
    chunksize: int = 50000,
    nrows: int | None = None,
    timestamp_col: str | None = None,
    strict_timestamps: bool = False,
) -> pd.DataFrame:
    """High-level loader for network traffic datasets.

    Loads CSV in chunks, standardizes column headers, parses timestamps with stable sorting,
    records source provenance, and returns a consolidated DataFrame.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if dataset_name == "auto":
        lower_name = path.name.lower()
        if "unsw" in lower_name:
            dataset_name = "unsw_nb15"
        elif "cic" in lower_name or "ids" in lower_name:
            dataset_name = "cic_ids2017"
        else:
            dataset_name = "generic"

    chunks = []
    for chunk in load_csv_chunks(path, chunksize=chunksize, nrows=nrows):
        chunks.append(chunk)

    if not chunks:
        return pd.DataFrame()

    combined = pd.concat(chunks, ignore_index=True)
    combined = parse_timestamps(
        combined,
        timestamp_col=timestamp_col,
        sort_chronological=True,
        strict=strict_timestamps,
    )
    combined = record_source_provenance(combined, dataset_name=dataset_name, filepath=path)
    return combined
