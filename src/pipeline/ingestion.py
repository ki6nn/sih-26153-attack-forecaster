"""Dataset ingestion module for loading and standardizing raw network traffic CSVs."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Generator, Sequence

import pandas as pd

# Explicit official dataset specifications
DATASET_SPECS: Final[dict[str, dict[str, Any]]] = {
    "unsw_nb15": {
        "dataset_name": "unsw_nb15",
        "official_name": "UNSW-NB15",
        "institution": "University of New South Wales (ADFA) / ACCS",
        "official_url": "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
        "timestamp_col": "stime",
        "duration_col": "dur",
        "duration_unit": "s",
        "precision": "1s_epoch",
        "description": "UNSW-NB15 raw captures; stime epoch seconds, dur in floating-point seconds.",
    },
    "cic_ids2017": {
        "dataset_name": "cic_ids2017",
        "official_name": "CIC-IDS2017",
        "institution": "Canadian Institute for Cybersecurity (UNB)",
        "official_url": "https://www.unb.ca/cic/datasets/ids-2017.html",
        "timestamp_col": "timestamp",
        "duration_col": "flow_duration",
        "duration_unit": "us",
        "precision": "1s_string",
        "description": "CIC-IDS2017 captures; string timestamps, flow_duration in microseconds.",
    },
}


@dataclass
class DatasetProvenance:
    """Explicit metadata establishing dataset origin, verification status, and research permissions."""

    dataset_name: str
    provenance_type: str  # "official_research", "synthetic_demonstration_data", "unit_test_fixture", "unverified"
    source_url: str | None = None
    dataset_version: str | None = None
    filename: str = ""
    sha256: str | None = None
    attribution: str | None = None
    license: str | None = None
    is_research_evidence: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert provenance record to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetProvenance":
        """Instantiate provenance record from dictionary."""
        valid_fields = {
            "dataset_name",
            "provenance_type",
            "source_url",
            "dataset_version",
            "filename",
            "sha256",
            "attribution",
            "license",
            "is_research_evidence",
            "metadata",
        }
        # Adapt keys from legacy schema if needed
        d = dict(data)
        if "file" in d and "filename" not in d:
            d["filename"] = Path(d["file"]).name
        if "source" in d and "source_url" not in d:
            d["source_url"] = str(d["source"])
        if "dataset_name" not in d:
            d["dataset_name"] = "generic"

        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    def save(self, filepath: str | Path) -> str:
        """Serialize provenance metadata to JSON."""
        dest = Path(filepath)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return str(dest)

    @classmethod
    def load(cls, filepath: str | Path) -> "DatasetProvenance":
        """Load provenance record from JSON."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Provenance record not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def compute_file_sha256(filepath: str | Path) -> str:
    """Compute cryptographic SHA-256 hash of a file."""
    path = Path(filepath)
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def verify_dataset_provenance(
    filepath: str | Path,
    provenance: DatasetProvenance | dict[str, Any] | str | Path | None = None,
    is_research: bool = False,
    enforce_provenance: bool = False,
) -> DatasetProvenance:
    """Verify dataset provenance, cryptographic checksum, and research eligibility.

    Policy:
        - Never infers official status merely from filenames containing 'cic' or 'unsw'.
        - If is_research=True or enforce_provenance=True:
            * Must have a verified provenance record with is_research_evidence=True.
            * provenance_type must be 'official_research'.
            * dataset_name must be an approved research dataset ('cic_ids2017', 'unsw_nb15').
            * If sha256 is present, it must match the computed file hash.
            * Synthetic demo data and unit test fixtures are strictly rejected for research.
        - If is_research=False:
            * Allows synthetic demo data and unit test fixtures for offline UI/smoke tests,
              stamped with is_research_evidence=False.

    Raises:
        ValueError: If provenance verification fails or synthetic data enters research mode.
        FileNotFoundError: If filepath does not exist.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path}")

    # Resolve provenance object
    prov_obj: DatasetProvenance | None = None
    if isinstance(provenance, DatasetProvenance):
        prov_obj = provenance
    elif isinstance(provenance, dict):
        prov_obj = DatasetProvenance.from_dict(provenance)
    elif isinstance(provenance, (str, Path)):
        prov_obj = DatasetProvenance.load(provenance)
    else:
        # Check standard sidecar provenance locations
        candidates = [
            path.with_name(f"{path.stem}_provenance.json"),
            path.with_name(f"{path.name}.provenance.json"),
            path.parent / f"{path.stem}_provenance.json",
        ]
        # In-repo sample demo provenance fallback
        if path.name == "sample_demo.csv":
            candidates.append(path.parent / "sample_demo_provenance.json")

        for cand in candidates:
            if cand.exists():
                prov_obj = DatasetProvenance.load(cand)
                break

    if prov_obj is None:
        if is_research or enforce_provenance:
            raise ValueError(
                f"Dataset provenance could not be verified for '{path.name}'. In strict research "
                f"mode, an explicit provenance record with is_research_evidence=True is required. "
                f"Never claim a dataset is official without registered provenance."
            )
        # Unverified fallback for non-research legacy/demo mode
        prov_obj = DatasetProvenance(
            dataset_name="unverified",
            provenance_type="unverified",
            filename=path.name,
            is_research_evidence=False,
        )

    # Check cryptographic hash if provided
    if prov_obj.sha256 is not None:
        actual_hash = compute_file_sha256(path)
        expected_hash = str(prov_obj.sha256).strip().lower()
        if actual_hash != expected_hash:
            raise ValueError(
                f"Provenance SHA256 checksum mismatch for '{path.name}'. "
                f"Expected {expected_hash}, calculated {actual_hash}. File may be corrupted or modified."
            )

    # Research mode validation
    if is_research:
        if not prov_obj.is_research_evidence or prov_obj.provenance_type != "official_research":
            raise ValueError(
                f"Dataset '{path.name}' has provenance_type='{prov_obj.provenance_type}' "
                f"(is_research_evidence={prov_obj.is_research_evidence}) and is strictly rejected "
                f"for empirical research under SIH 26153 rules. Synthetic demonstration data and "
                f"unverified extracts cannot be used as research evidence."
            )
        if prov_obj.dataset_name not in DATASET_SPECS:
            raise ValueError(
                f"Dataset identity '{prov_obj.dataset_name}' is not recognized as an approved official "
                f"research benchmark. Approved datasets: {list(DATASET_SPECS.keys())}."
            )
        if not prov_obj.source_url:
            raise ValueError(
                f"Dataset provenance for '{prov_obj.dataset_name}' is missing required official source URL."
            )

    return prov_obj


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
    if strict and nat_count > 0:
        raw_null_count = int(raw_series.isna().sum())
        unparseable_count = nat_count - raw_null_count
        raise ValueError(
            f"Strict timestamp parsing failure for column '{detected_col}': "
            f"{nat_count} invalid timestamps ({raw_null_count} null, "
            f"{unparseable_count} unparseable). In strict research mode, all flow "
            f"timestamps must be fully observed and valid."
        )

    if sort_chronological and not result["datetime"].isna().all():
        # Stable sort preserves identical timestamp groupings in original relative order
        result = result.sort_values(by="datetime", ascending=True, kind="stable").reset_index(drop=True)

    return result


def record_source_provenance(
    df: pd.DataFrame,
    dataset_name: str,
    filepath: str | Path,
    is_research_evidence: bool = False,
) -> pd.DataFrame:
    """Attach dataset origin and file provenance metadata to the dataframe."""
    result = df.copy()
    result["_source_dataset"] = str(dataset_name)
    result["_source_file"] = Path(filepath).name
    result["_is_research_evidence"] = bool(is_research_evidence)
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
    enforce_provenance: bool = False,
    provenance: DatasetProvenance | dict[str, Any] | str | Path | None = None,
    is_research: bool = False,
) -> pd.DataFrame:
    """High-level loader for network traffic datasets.

    Loads CSV in chunks, standardizes column headers, verifies cryptographic provenance
    if requested or in research mode, parses timestamps with stable sorting,
    records source provenance, and returns a consolidated DataFrame.

    Raises:
        ValueError: If provenance verification fails, or timestamps are invalid in strict mode.
        FileNotFoundError: If filepath does not exist.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    prov_record: DatasetProvenance | None = None
    if is_research or enforce_provenance or provenance is not None:
        prov_record = verify_dataset_provenance(
            path,
            provenance=provenance,
            is_research=is_research,
            enforce_provenance=enforce_provenance,
        )
        if dataset_name == "auto" and prov_record.dataset_name != "generic":
            dataset_name = prov_record.dataset_name

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
        strict=strict_timestamps or is_research,
    )
    is_research_ev = prov_record.is_research_evidence if prov_record else False
    combined = record_source_provenance(
        combined,
        dataset_name=dataset_name,
        filepath=path,
        is_research_evidence=is_research_ev,
    )
    if prov_record is not None:
        combined.attrs["provenance"] = prov_record

    return combined
