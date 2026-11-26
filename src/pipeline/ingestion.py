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
) -> pd.DataFrame:
    """Identify and parse timestamp column into standardized UTC datetimes.

    Supports:
        - ISO datetime strings (e.g. '2017-07-07 09:30:00' or '07/07/2017 09:30:00')
        - Epoch unix timestamps in seconds or milliseconds (e.g. UNSW-NB15 stime/ltime)
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

    sample = result[detected_col].dropna().iloc[:10] if len(result[detected_col].dropna()) > 0 else []
    is_numeric = False
    try:
        pd.to_numeric(sample)
        is_numeric = True
    except (ValueError, TypeError):
        is_numeric = False

    if is_numeric:
        raw_numeric = pd.to_numeric(result[detected_col], errors="coerce")
        median_val = raw_numeric.median() if len(raw_numeric.dropna()) > 0 else 0
        if median_val > 1e11:
            unit = "ms"
        elif median_val > 1e8:
            unit = "s"
        else:
            unit = "s"
        result["datetime"] = pd.to_datetime(raw_numeric, unit=unit, utc=True, errors="coerce")
    else:
        result["datetime"] = pd.to_datetime(
            result[detected_col], utc=True, errors="coerce", format="mixed"
        )

    if sort_chronological and "datetime" in result.columns and not result["datetime"].isna().all():
        result = result.sort_values(by="datetime", ascending=True).reset_index(drop=True)

    return result


def record_source_provenance(
    df: pd.DataFrame,
    dataset_name: str,
    filepath: str | Path,
) -> pd.DataFrame:
    """Attach dataset origin and file provenance to the dataframe."""
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
) -> pd.DataFrame:
    """High-level loader for network traffic datasets.

    Loads CSV in chunks, standardizes columns, parses timestamps, adds provenance,
    and returns a consolidated, chronologically sorted DataFrame.
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
    combined = parse_timestamps(combined, timestamp_col=timestamp_col, sort_chronological=True)
    combined = record_source_provenance(combined, dataset_name=dataset_name, filepath=path)
    return combined
