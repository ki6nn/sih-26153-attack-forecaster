"""NetworkSentinel AI — Phase 2 integration entry point.

Runs the full leakage-safe research pipeline, trains the TemporalWorldModel
and reproducible baselines (LogisticRegression, RandomForest), evaluates all
three side-by-side across four forecast horizons, and prints a comparison table.

IMPORTANT — Dataset restrictions
---------------------------------
Official CIC-IDS2017 and UNSW-NB15 archives have NOT yet been downloaded.
This script uses only the existing demonstration slice (sample_demo.csv) or
synthetic fixtures.  Results printed here are DEVELOPMENT SMOKE-TEST outputs,
NOT research benchmarks.  Research benchmarking is pending official data
acquisition.  Do not interpret these numbers as performance claims.

Usage
-----
  python main.py                 # smoke test with demo data
  python main.py --research      # attempt research pipeline (needs official data)
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import numpy as np
import torch

from src.contracts import (
    ATTACK_CATEGORY_GROUPS,
    FORECAST_STEPS,
    NUM_FEATURES,
    WINDOW_SIZE,
    get_mock_forecast,
)
from src.model.baseline import (
    LogisticRegressionBaseline,
    RandomForestBaseline,
    evaluate_models,
)
from src.model.evaluation import (
    compare_all_models,
    evaluate_model_predictions,
    evaluate_state_reconstruction,
)
from src.model.world_model import (
    AttackForecasterNet,
    predict_window,
    train_temporal_world_model,
    train_world_model,
)
from src.pipeline.preprocessor import (
    FEATURE_COLUMNS,
    prepare_research_pipeline,
    preprocess,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def _print_horizon_table(res: dict, model_name: str) -> None:
    """Print a compact per-horizon metric table for one model."""
    print(f"\n  [{model_name}]")
    print(f"  {'Horizon':<10}{'F1':>8}{'Precision':>12}{'Recall':>10}{'FPR':>8}{'PR-AUC':>9}{'Brier':>8}")
    print(f"  {'-'*65}")
    for step in range(1, FORECAST_STEPS + 1):
        h_key = f"horizon_{step}"
        if h_key not in res:
            continue
        m = res[h_key]
        print(
            f"  +{step} flow(s) {m['f1']:>8.4f}{m['precision']:>12.4f}"
            f"{m['recall']:>10.4f}{m['fpr']:>8.4f}{m['pr_auc']:>9.4f}{m['brier_score']:>8.4f}"
        )
    om = res.get("overall_mean", {})
    print(
        f"  {'Mean':<10} {om.get('mean_f1',0):.4f}"
        f"{'':>12}{om.get('mean_brier',0):>18.4f}"
    )


# ---------------------------------------------------------------------------
# Smoke-test pipeline (uses demo/synthetic data)
# ---------------------------------------------------------------------------

def run_smoke_test() -> None:
    """Run a development smoke test using synthetic windows.

    No official datasets required.  Results are sanity checks only.
    """
    _print_section("NetworkSentinel AI — Phase 2 Development Smoke Test")
    print(textwrap.dedent("""
    WARNING: Official CIC-IDS2017 and UNSW-NB15 datasets are not yet available.
    Training on SYNTHETIC FIXTURES only.  These numbers are NOT research results.
    """))

    rng = np.random.default_rng(42)
    n_tr, n_vl, n_te = 128, 32, 32

    X_tr  = rng.standard_normal((n_tr, WINDOW_SIZE, NUM_FEATURES)).astype(np.float32)
    y_tr  = rng.binomial(1, 0.3, (n_tr, FORECAST_STEPS)).astype(np.float32)
    Xt_tr = rng.standard_normal((n_tr, FORECAST_STEPS, NUM_FEATURES)).astype(np.float32)
    yc_tr = rng.integers(0, len(ATTACK_CATEGORY_GROUPS), (n_tr, FORECAST_STEPS)).astype(np.int64)

    X_vl  = rng.standard_normal((n_vl, WINDOW_SIZE, NUM_FEATURES)).astype(np.float32)
    y_vl  = rng.binomial(1, 0.3, (n_vl, FORECAST_STEPS)).astype(np.float32)
    Xt_vl = rng.standard_normal((n_vl, FORECAST_STEPS, NUM_FEATURES)).astype(np.float32)
    yc_vl = rng.integers(0, len(ATTACK_CATEGORY_GROUPS), (n_vl, FORECAST_STEPS)).astype(np.int64)

    X_te  = rng.standard_normal((n_te, WINDOW_SIZE, NUM_FEATURES)).astype(np.float32)
    y_te  = rng.binomial(1, 0.3, (n_te, FORECAST_STEPS)).astype(np.float32)
    Xt_te = rng.standard_normal((n_te, FORECAST_STEPS, NUM_FEATURES)).astype(np.float32)
    yc_te = rng.integers(0, len(ATTACK_CATEGORY_GROUPS), (n_te, FORECAST_STEPS)).astype(np.int64)

    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)

    # --- Temporal World Model ---
    print("Training TemporalWorldModel (CPU, 5 epochs, synthetic)…")
    world_model, wm_history = train_temporal_world_model(
        X_train=X_tr,  X_target_train=Xt_tr,
        y_risk_train=y_tr, y_cat_train=yc_tr,
        X_val=X_vl,   X_target_val=Xt_vl,
        y_risk_val=y_vl, y_cat_val=yc_vl,
        epochs=5, batch_size=32,
        save_path=artifacts / "world_model_smoke.pt",
        device="cpu",
    )
    final_ep = wm_history["history"][-1]
    print(
        f"  Epoch {final_ep['epoch']} — train_loss={final_ep['train_loss']:.4f} "
        f"val_loss={final_ep['val_loss']:.4f} "
        f"tf_prob={final_ep['teacher_forcing_prob']:.2f}"
    )

    # World model inference on test set
    world_model.eval()
    with torch.no_grad():
        wm_states, wm_risk_logits, _ = world_model(
            torch.from_numpy(X_te)
        )
    wm_probs = torch.sigmoid(wm_risk_logits).cpu().numpy()       # (N_te, 4)
    wm_state_preds = wm_states.cpu().numpy()                      # (N_te, 4, 20)

    # --- Logistic Regression Baseline ---
    print("Training LogisticRegressionBaseline…")
    lr = LogisticRegressionBaseline(forecast_steps=FORECAST_STEPS, random_state=42)
    lr.fit(X_tr, y_tr)
    lr_probs = lr.predict_proba(X_te)                             # (N_te, 4)

    # --- Random Forest Baseline ---
    print("Training RandomForestBaseline (n_estimators=50)…")
    rf = RandomForestBaseline(n_estimators=50, forecast_steps=FORECAST_STEPS, random_state=42)
    rf.fit(X_tr, y_tr)
    rf_probs = rf.predict_risk_proba(X_te)                        # (N_te, 4)

    # --- Evaluation ---
    _print_section("Multi-horizon Evaluation (synthetic test set — not research results)")

    all_res, summary_df = compare_all_models(wm_probs, lr_probs, rf_probs, y_te)

    for name, key in [
        ("Temporal World Model", "world_model"),
        ("Logistic Regression",  "logistic_regression"),
        ("Random Forest",        "random_forest"),
    ]:
        _print_horizon_table(all_res[key], name)

    # State reconstruction (World Model only)
    state_res = evaluate_state_reconstruction(Xt_te, wm_state_preds)
    print(f"\n  World Model State Reconstruction — overall MSE: {state_res['overall_mse']:.6f} "
          f"MAE: {state_res['overall_mae']:.6f}")

    # --- Live single-window forecast example ---
    _print_section("Live Single-Window Forecast Demo")
    feature_names = FEATURE_COLUMNS if len(FEATURE_COLUMNS) == NUM_FEATURES else [f"feat_{i}" for i in range(NUM_FEATURES)]
    sample_window = X_te[0]  # shape (W, D)
    forecast = predict_window(world_model, sample_window, feature_names)
    print(f"  Current stage   : {forecast.current_stage}")
    print(f"  Next-step group : {forecast.predicted_next_stage}")
    print(f"  Horizon risks   : {[f'{p:.3f}' for p in forecast.forecast_probs]}")
    print(f"  Top features    : {[f['feature'] for f in forecast.top_features[:3]]}")

    _print_section("Artifacts")
    for artefact in artifacts.glob("*.pt"):
        print(f"  {artefact}")
    for artefact in artifacts.glob("*.pkl"):
        print(f"  {artefact}")

    print("\n[SMOKE TEST COMPLETE]  All models trained and evaluated successfully.")
    print("NOTE: Research benchmarking pending official CIC-IDS2017 / UNSW-NB15 acquisition.")


# ---------------------------------------------------------------------------
# Research pipeline (requires official datasets)
# ---------------------------------------------------------------------------

def run_research_pipeline() -> None:
    """Attempt full research pipeline with officially acquired datasets.

    Will raise if official data is unavailable.
    """
    _print_section("NetworkSentinel AI — Research Pipeline")

    data = prepare_research_pipeline(enforce_provenance=True)
    print(f"  Dataset        : {data.dataset_name}")
    print(f"  Feature schema : {data.feature_names}")
    print(f"  Train windows  : {data.X_train.shape}")
    print(f"  Val windows    : {data.X_val.shape}")
    print(f"  Test windows   : {data.X_test.shape}")

    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)

    print("\nTraining TemporalWorldModel (research data)…")
    world_model, wm_history = train_temporal_world_model(
        X_train=data.X_train,         X_target_train=data.X_target_train,
        y_risk_train=data.y_train,    y_cat_train=data.y_category_train,
        X_val=data.X_val,             X_target_val=data.X_target_val,
        y_risk_val=data.y_val,        y_cat_val=data.y_category_val,
        epochs=15, batch_size=64,
        save_path=artifacts / "world_model.pt",
        device="cpu",
    )
    best_val = wm_history.get("best_val_loss", "N/A")
    print(f"  Best validation loss: {best_val}")

    print("\nTraining baseline models…")
    lr = LogisticRegressionBaseline(forecast_steps=FORECAST_STEPS, random_state=42)
    lr.fit(data.X_train, data.y_train)

    rf = RandomForestBaseline(n_estimators=100, forecast_steps=FORECAST_STEPS, random_state=42)
    rf.fit(data.X_train, data.y_train)

    world_model.eval()
    with torch.no_grad():
        wm_states, wm_risk_logits, _ = world_model(torch.from_numpy(data.X_test))
    wm_probs = torch.sigmoid(wm_risk_logits).cpu().numpy()
    wm_state_preds = wm_states.cpu().numpy()

    lr_probs = lr.predict_proba(data.X_test)
    rf_probs = rf.predict_risk_proba(data.X_test)

    _print_section("Multi-horizon Evaluation (official test set)")
    all_res, summary_df = compare_all_models(wm_probs, lr_probs, rf_probs, data.y_test)
    for name, key in [
        ("Temporal World Model", "world_model"),
        ("Logistic Regression",  "logistic_regression"),
        ("Random Forest",        "random_forest"),
    ]:
        _print_horizon_table(all_res[key], name)

    state_res = evaluate_state_reconstruction(data.X_target_test, wm_state_preds)
    print(f"\n  State MSE: {state_res['overall_mse']:.6f}  MAE: {state_res['overall_mae']:.6f}")

    lr.save(str(artifacts / "baseline_lr.pkl"))
    rf.save(str(artifacts / "baseline_rf.pkl"))
    print("\n[RESEARCH PIPELINE COMPLETE]")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NetworkSentinel AI Phase 2 — Temporal World Model Integration"
    )
    parser.add_argument(
        "--research",
        action="store_true",
        default=False,
        help="Run research pipeline (requires official CIC-IDS2017 / UNSW-NB15 datasets).",
    )
    args = parser.parse_args()

    if args.research:
        run_research_pipeline()
    else:
        run_smoke_test()


if __name__ == "__main__":
    main()
