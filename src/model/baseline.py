from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.contracts import MITRE_STAGE_MAP, NUM_FEATURES


class BaselineAttackClassifier:
    """Random Forest baseline for attack detection and next-stage prediction."""

    def __init__(
        self,
        n_estimators: int = 200,
        random_state: int = 42,
        save_path: str = "artifacts/baseline_rf.pkl",
    ) -> None:
        self.save_path = save_path
        self.attack_model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
        self.stage_model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
        self.is_fitted = False
        self.stage_names = list(MITRE_STAGE_MAP)
        self.stage_lookup = {name: index for index, name in enumerate(self.stage_names)}

    @staticmethod
    def _prepare_features(features) -> np.ndarray:
        array = np.asarray(features)
        if array.ndim == 3:
            if array.shape[1:] != (10, NUM_FEATURES):
                raise ValueError(
                    f"Three-dimensional features must have shape (N, 10, {NUM_FEATURES})"
                )
            array = array[:, -1, :]
        elif array.ndim == 2 and array.shape[1] == NUM_FEATURES:
            pass
        else:
            array = array.reshape(array.shape[0], -1)

        if array.shape[1] != NUM_FEATURES:
            raise ValueError(
                f"Features must contain {NUM_FEATURES} values per sample; got {array.shape[1]}"
            )
        return array.astype(np.float32)

    @staticmethod
    def _encode_stage_labels(y_stage: Sequence) -> np.ndarray:
        stage_names = list(MITRE_STAGE_MAP)
        lookup = {name.lower(): index for index, name in enumerate(stage_names)}
        encoded = []
        for stage in np.asarray(y_stage).reshape(-1):
            if isinstance(stage, (int, np.integer)):
                index = int(stage)
            else:
                index = lookup.get(str(stage).lower())
                if index is None:
                    raise ValueError(f"Unknown MITRE stage: {stage}")
            if index < 0 or index >= len(stage_names):
                raise ValueError(f"Stage index out of range: {index}")
            encoded.append(index)
        return np.asarray(encoded, dtype=np.int64)

    def fit(
        self,
        X_train,
        y_attack,
        y_stage=None,
        save_path: str | None = None,
    ) -> "BaselineAttackClassifier":
        features = self._prepare_features(X_train)
        attack_labels = np.asarray(y_attack).reshape(-1)
        if features.shape[0] != attack_labels.shape[0]:
            raise ValueError("X_train and y_attack must contain the same number of samples")
        if set(np.unique(attack_labels).tolist()) - {0, 1}:
            raise ValueError("y_attack must contain binary labels 0 and 1")

        self.attack_model.fit(features, attack_labels)

        if y_stage is not None:
            stage_labels = self._encode_stage_labels(y_stage)
            if stage_labels.shape[0] != features.shape[0]:
                raise ValueError("X_train and y_stage must contain the same number of samples")
            self.stage_model.fit(features, stage_labels)

        self.is_fitted = True
        destination = Path(save_path or self.save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)
        return self

    def predict_attack(self, features) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("The baseline classifier has not been fitted")
        return self.attack_model.predict(self._prepare_features(features)).astype(int)

    def predict_attack_proba(self, features) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("The baseline classifier has not been fitted")
        probabilities = self.attack_model.predict_proba(self._prepare_features(features))
        if probabilities.shape[1] == 2:
            return probabilities[:, 1]
        return probabilities[:, 0]

    def predict_stage(self, features) -> np.ndarray:
        if not hasattr(self.stage_model, "classes_"):
            raise RuntimeError("The next-stage model has not been fitted")
        indices = self.stage_model.predict(self._prepare_features(features)).astype(int)
        return np.asarray([self.stage_names[index] for index in indices])

    def predict_stage_indices(self, features) -> np.ndarray:
        if not hasattr(self.stage_model, "classes_"):
            raise RuntimeError("The next-stage model has not been fitted")
        return self.stage_model.predict(self._prepare_features(features)).astype(int)

    def save(self, save_path: str | None = None) -> str:
        destination = Path(save_path or self.save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)
        return str(destination)

    @staticmethod
    def load(save_path: str = "artifacts/baseline_rf.pkl") -> "BaselineAttackClassifier":
        return joblib.load(save_path)


def _binary_probabilities(probabilities) -> np.ndarray:
    array = np.asarray(probabilities, dtype=np.float64)
    if array.ndim == 1:
        return np.clip(array, 0.0, 1.0)
    if array.ndim != 2:
        raise ValueError("Probability inputs must be one- or two-dimensional")

    if array.shape[1] == 1:
        return np.clip(array[:, 0], 0.0, 1.0)
    if array.shape[1] == 2:
        return np.clip(array[:, 1], 0.0, 1.0)
    return np.clip(np.max(array, axis=1), 0.0, 1.0)


def evaluate_models(
    world_model_probs,
    baseline_probs,
    y_true,
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    truth = np.asarray(y_true).reshape(-1)
    unique_truth = np.unique(truth)
    if not set(unique_truth.tolist()).issubset({0, 1}):
        raise ValueError("y_true must contain binary labels 0 and 1")

    world_probabilities = _binary_probabilities(world_model_probs)
    baseline_probabilities = _binary_probabilities(baseline_probs)
    if world_probabilities.shape[0] != truth.shape[0]:
        raise ValueError("world_model_probs and y_true must have the same sample count")
    if baseline_probabilities.shape[0] != truth.shape[0]:
        raise ValueError("baseline_probs and y_true must have the same sample count")

    comparison: dict[str, dict[str, float]] = {}
    for model_name, probabilities in (
        ("World Model", world_probabilities),
        ("Baseline RF", baseline_probabilities),
    ):
        predictions = (probabilities >= 0.5).astype(int)
        auc = roc_auc_score(truth, probabilities) if len(unique_truth) == 2 else 0.0
        comparison[model_name] = {
            "Accuracy": float(accuracy_score(truth, predictions)),
            "Precision": float(precision_score(truth, predictions, zero_division=0)),
            "Recall": float(recall_score(truth, predictions, zero_division=0)),
            "F1-Score": float(f1_score(truth, predictions, zero_division=0)),
            "ROC-AUC": float(auc),
        }

    summary = pd.DataFrame(comparison).T
    summary.index.name = "Model"
    return comparison, summary


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    X = rng.normal(size=(120, NUM_FEATURES))
    y_attack = (X[:, 0] + X[:, 1] > 0).astype(int)
    y_stage = np.where(y_attack == 0, "Normal", "Exploits")

    baseline = BaselineAttackClassifier()
    baseline.fit(X, y_attack, y_stage)
    predictions = baseline.predict_attack(X[:10])
    stage_predictions = baseline.predict_stage(X[:10])

    loaded = BaselineAttackClassifier.load(baseline.save_path)
    loaded_predictions = loaded.predict_attack(X[:10])

    world_probabilities = np.clip(y_attack * 0.8 + 0.15 + rng.normal(0, 0.08, 120), 0, 1)
    baseline_probabilities = loaded.predict_attack_proba(X)
    comparison, summary = evaluate_models(
        world_probabilities,
        baseline_probabilities,
        y_attack,
    )

    print("Attack predictions:", predictions.tolist())
    print("Stage predictions:", stage_predictions.tolist())
    print("Loaded predictions match:", np.array_equal(predictions, loaded_predictions[:10]))
    print(summary.to_string(float_format=lambda value: f"{value:.4f}"))
