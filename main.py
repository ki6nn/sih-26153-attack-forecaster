"""Train and benchmark the SIH 2026 attack forecasting models."""

from pathlib import Path

import numpy as np
import torch

from src.contracts import get_mock_forecast
from src.model.baseline import BaselineAttackClassifier, evaluate_models
from src.model.world_model import AttackForecasterNet, predict_window, train_world_model
from src.pipeline.preprocessor import FEATURE_COLUMNS, preprocess


def main() -> None:
    X, y_probs, y_stage = preprocess()
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_probs_train, y_probs_test = y_probs[:split], y_probs[split:]
    y_stage_train, y_stage_test = y_stage[:split], y_stage[split:]

    print("Training World Model...")
    world_model = train_world_model(
        X_train,
        y_probs_train,
        y_stage_train,
        epochs=15,
        save_path="artifacts/world_model.pt",
    )

    baseline = BaselineAttackClassifier(save_path="artifacts/baseline_rf.pkl")
    baseline.fit(X_train, y_probs_train[:, 0], y_stage_train)

    world_model.eval()
    with torch.no_grad():
        world_probs = world_model(torch.from_numpy(X_test.astype(np.float32)))[0].cpu().numpy()
    baseline_probs = baseline.predict_attack_proba(X_test)

    _, summary = evaluate_models(
        world_probs[:, 0],
        baseline_probs,
        y_probs_test[:, 0],
    )
    print("\nSide-by-side benchmark")
    print(summary.to_string(float_format=lambda value: f"{value:.4f}"))

    if len(X_test):
        live_forecast = predict_window(world_model, X_test[-1], FEATURE_COLUMNS)
    else:
        live_forecast = get_mock_forecast()
    print("\nLive sample ForecastOutput")
    print(live_forecast.to_dict())

    print("\nArtifacts:")
    print(f"- {Path('artifacts/world_model.pt')}")
    print(f"- {Path('artifacts/baseline_rf.pkl')}")


if __name__ == "__main__":
    main()
