"""Prepare UNSW-NB15 flows for the attack forecaster."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.contracts import (
    FORECAST_STEPS,
    MITRE_STAGE_MAP,
    NUM_FEATURES,
    WINDOW_SIZE,
)


FEATURE_COLUMNS: Final[list[str]] = [
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

_STAGES: Final[list[str]] = [
    "Normal",
    "Reconnaissance",
    "Exploits",
    "Backdoor",
    "DoS",
]
_DATA_FILES: Final[tuple[Path, Path]] = (
    Path("data/UNSW_NB15_training-set.csv"),
    Path("data/UNSW_NB15_testing-set.csv"),
)


def _synthetic_dataset(rows_per_stage: int = 120) -> pd.DataFrame:
    """Create deterministic, stage-shaped flows when the downloaded data is absent."""
    rng = np.random.default_rng(42)
    records: list[dict[str, object]] = []
    for stage_index, stage in enumerate(_STAGES):
        for _ in range(rows_per_stage):
            intensity = float(stage_index)
            values = {
                "dur": max(0.001, rng.lognormal(-1.0 + intensity * 0.18, 0.35)),
                "spkts": max(1.0, rng.poisson(8 + intensity * 10)),
                "dpkts": max(1.0, rng.poisson(7 + intensity * 8)),
                "sbytes": max(40.0, rng.lognormal(7.0 + intensity * 0.35, 0.45)),
                "dbytes": max(40.0, rng.lognormal(7.1 + intensity * 0.3, 0.45)),
                "rate": max(0.1, rng.lognormal(3.0 + intensity * 0.55, 0.35)),
                "sttl": np.clip(rng.normal(120 - intensity * 8, 10), 1, 255),
                "dttl": np.clip(rng.normal(115 - intensity * 7, 10), 1, 255),
                "sload": max(1.0, rng.lognormal(8.0 + intensity * 0.45, 0.4)),
                "dload": max(1.0, rng.lognormal(7.8 + intensity * 0.4, 0.4)),
                "sloss": max(0.0, rng.poisson(intensity * 1.5)),
                "dloss": max(0.0, rng.poisson(intensity * 1.2)),
                "sinpkt": max(0.01, rng.lognormal(2.0 - intensity * 0.18, 0.3)),
                "dinpkt": max(0.01, rng.lognormal(2.1 - intensity * 0.16, 0.3)),
                "sjit": max(0.0, rng.lognormal(-0.2 + intensity * 0.2, 0.35)),
                "djit": max(0.0, rng.lognormal(-0.25 + intensity * 0.2, 0.35)),
                "swin": max(0.0, rng.normal(6000 - intensity * 600, 500)),
                "dwin": max(0.0, rng.normal(6000 - intensity * 550, 500)),
                "ct_state_ttl": np.clip(rng.normal(2 + intensity, 0.5), 0, 10),
                "ct_srv_src": max(1.0, rng.poisson(3 + intensity * 5)),
                "attack_cat": stage,
            }
            records.append(values)
    return pd.DataFrame.from_records(records, columns=[*FEATURE_COLUMNS, "attack_cat"])


def _load_dataset() -> pd.DataFrame:
    present_files = [path for path in _DATA_FILES if path.exists()]
    if not present_files:
        return _synthetic_dataset()

    frames = [pd.read_csv(path) for path in present_files]
    data = pd.concat(frames, ignore_index=True)
    missing = [column for column in FEATURE_COLUMNS if column not in data.columns]
    if "attack_cat" not in data.columns:
        raise ValueError("UNSW-NB15 CSV must contain an 'attack_cat' column")
    if missing:
        raise ValueError(f"UNSW-NB15 CSV is missing features: {', '.join(missing)}")
    return data


def _stage_name(value: object) -> str:
    normalized = str(value).strip().lower()
    for stage in MITRE_STAGE_MAP:
        if stage.lower() == normalized:
            return stage
    if normalized in {"benign", "normal", "nan", ""}:
        return "Normal"
    raise ValueError(f"Unsupported attack_cat label: {value!r}")


def preprocess(
    data: pd.DataFrame | None = None,
    scaler_path: str | Path = "artifacts/scaler.pkl",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale flows and return ``(X, y_probs, y_stage)`` training arrays."""
    frame = _load_dataset() if data is None else data.copy()
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing or "attack_cat" not in frame.columns:
        raise ValueError("Input data must contain all 20 numeric features and attack_cat")

    numeric = frame[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    stages = frame["attack_cat"].map(_stage_name).to_numpy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric).astype(np.float32)

    destination = Path(scaler_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, destination)

    sample_count = len(frame) - WINDOW_SIZE - FORECAST_STEPS + 1
    if sample_count <= 0:
        raise ValueError(
            f"At least {WINDOW_SIZE + FORECAST_STEPS} chronological rows are required"
        )
    X = np.stack(
        [scaled[index : index + WINDOW_SIZE] for index in range(sample_count)]
    )
    future_stages = np.stack(
        [
            stages[index + WINDOW_SIZE : index + WINDOW_SIZE + FORECAST_STEPS]
            for index in range(sample_count)
        ]
    )
    y_probs = (future_stages != "Normal").astype(np.float32)
    stage_indices = {stage: index for index, stage in enumerate(MITRE_STAGE_MAP)}
    y_stage = np.asarray(
        [stage_indices[future[0]] for future in future_stages], dtype=np.int64
    )
    return X, y_probs, y_stage


def export_demo_sample(
    out_path: str | Path = "data/sample_demo.csv", rows: int = 500
) -> None:
    """Export a chronological multi-stage flow sample for offline demonstrations."""
    if rows < 1:
        raise ValueError("rows must be positive")
    demo = _synthetic_dataset(rows_per_stage=max(1, int(np.ceil(rows / len(_STAGES)))))
    demo = demo.iloc[:rows].copy()
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    demo.to_csv(destination, index=False)


if __name__ == "__main__":
    X, y_probs, y_stage = preprocess()
    export_demo_sample()
    print(f"Prepared X={X.shape}, y_probs={y_probs.shape}, y_stage={y_stage.shape}")
    print("Saved artifacts/scaler.pkl and data/sample_demo.csv")
