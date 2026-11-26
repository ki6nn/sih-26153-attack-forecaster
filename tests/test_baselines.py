"""Tests for LogisticRegressionBaseline, RandomForestBaseline, and evaluation engine.

All tests use CPU-only synthetic fixtures. No official datasets are used.
Benchmark results on CIC-IDS2017 and UNSW-NB15 are pending official data acquisition.
"""

import tempfile
import unittest
import numpy as np

from src.contracts import FEATURE_DIM, FORECAST_STEPS, NUM_ATTACK_CATEGORIES, WINDOW_SIZE, NUM_FEATURES
from src.model.baseline import (
    BaselineAttackClassifier,
    LogisticRegressionBaseline,
    RandomForestBaseline,
)
from src.model.evaluation import (
    compare_all_models,
    evaluate_model_predictions,
    evaluate_state_reconstruction,
)


def _synthetic_data(n: int, seed: int = 42):
    """Return (X, y_risk) arrays of shape (n, W, D) and (n, H)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, WINDOW_SIZE, NUM_FEATURES)).astype(np.float32)
    y = rng.binomial(1, 0.4, (n, FORECAST_STEPS)).astype(np.float32)
    return X, y


class TestLogisticRegressionBaseline(unittest.TestCase):
    """Unit tests for LogisticRegressionBaseline."""

    def setUp(self):
        self.X_train, self.y_train = _synthetic_data(60, seed=0)
        self.X_test,  self.y_test  = _synthetic_data(20, seed=1)

    def test_shapes_and_probability_range(self):
        """fit + predict_proba must return (N, H) probabilities in [0, 1]."""
        lr = LogisticRegressionBaseline(forecast_steps=FORECAST_STEPS, random_state=42)
        lr.fit(self.X_train, self.y_train)

        probs = lr.predict_proba(self.X_test)
        self.assertEqual(probs.shape, (len(self.X_test), FORECAST_STEPS))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

    def test_predict_binary(self):
        """predict must return 0/1 integers of shape (N, H)."""
        lr = LogisticRegressionBaseline(forecast_steps=FORECAST_STEPS, random_state=42)
        lr.fit(self.X_train, self.y_train)

        preds = lr.predict(self.X_test, threshold=0.5)
        self.assertEqual(preds.shape, (len(self.X_test), FORECAST_STEPS))
        self.assertTrue(np.all((preds == 0) | (preds == 1)))

    def test_all_benign_training_set(self):
        """Single-class training (all zeros) must not raise; probs must be constant."""
        y_zeros = np.zeros((60, FORECAST_STEPS), dtype=np.float32)
        lr = LogisticRegressionBaseline(forecast_steps=FORECAST_STEPS, random_state=42)
        lr.fit(self.X_train, y_zeros)

        probs = lr.predict_proba(self.X_test)
        self.assertEqual(probs.shape, (len(self.X_test), FORECAST_STEPS))
        # All-benign training → all predictions 0.0
        np.testing.assert_allclose(probs, 0.0)


class TestRandomForestBaseline(unittest.TestCase):
    """Unit tests for RandomForestBaseline."""

    def setUp(self):
        self.X_train, self.y_train = _synthetic_data(60, seed=2)
        self.X_test,  self.y_test  = _synthetic_data(20, seed=3)

    def test_shapes_and_probability_range(self):
        """fit + predict_risk_proba must return (N, H) probabilities in [0, 1]."""
        rf = RandomForestBaseline(n_estimators=5, forecast_steps=FORECAST_STEPS, random_state=42)
        rf.fit(self.X_train, self.y_train)

        probs = rf.predict_risk_proba(self.X_test)
        self.assertEqual(probs.shape, (len(self.X_test), FORECAST_STEPS))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

    def test_predict_proba_alias(self):
        """predict_proba must be an alias for predict_risk_proba."""
        rf = RandomForestBaseline(n_estimators=5, forecast_steps=FORECAST_STEPS, random_state=42)
        rf.fit(self.X_train, self.y_train)

        p1 = rf.predict_risk_proba(self.X_test)
        p2 = rf.predict_proba(self.X_test)
        np.testing.assert_array_equal(p1, p2)

    def test_predict_binary(self):
        """predict must return 0/1 integers of shape (N, H)."""
        rf = RandomForestBaseline(n_estimators=5, forecast_steps=FORECAST_STEPS, random_state=42)
        rf.fit(self.X_train, self.y_train)

        preds = rf.predict(self.X_test, threshold=0.5)
        self.assertEqual(preds.shape, (len(self.X_test), FORECAST_STEPS))
        self.assertTrue(np.all((preds == 0) | (preds == 1)))

    def test_deterministic_reproducibility(self):
        """Same random_state must produce bitwise-identical probabilities."""
        def run(seed):
            rf = RandomForestBaseline(n_estimators=5, forecast_steps=FORECAST_STEPS, random_state=seed)
            rf.fit(self.X_train, self.y_train)
            return rf.predict_risk_proba(self.X_test)

        np.testing.assert_array_equal(run(123), run(123))


class TestBaselineAttackClassifier(unittest.TestCase):
    """Backward-compatibility adapter tests."""

    def setUp(self):
        self.X_train, self.y_train = _synthetic_data(40, seed=5)
        self.X_test, _ = _synthetic_data(10, seed=6)

    def test_fit_and_predict(self):
        """BaselineAttackClassifier must fit and produce per-step binary predictions."""
        clf = BaselineAttackClassifier(n_estimators=5)
        clf.fit(self.X_train, self.y_train)

        preds = clf.predict(self.X_test)
        self.assertEqual(preds.shape, (len(self.X_test), FORECAST_STEPS))

    def test_predict_risk_proba(self):
        """predict_risk_proba must return valid probabilities."""
        clf = BaselineAttackClassifier(n_estimators=5)
        clf.fit(self.X_train, self.y_train)

        probs = clf.predict_risk_proba(self.X_test)
        self.assertEqual(probs.shape, (len(self.X_test), FORECAST_STEPS))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))


class TestEvaluationEngine(unittest.TestCase):
    """Unit tests for evaluation metrics and comparisons."""

    def setUp(self):
        rng = np.random.default_rng(99)
        self.n = 50
        self.h = FORECAST_STEPS
        self.d = NUM_FEATURES

        self.y_true = rng.binomial(1, 0.3, (self.n, self.h)).astype(np.float32)
        self.y_prob = rng.uniform(0.0, 1.0, (self.n, self.h)).astype(np.float32)

    def test_evaluate_model_predictions_structure(self):
        """evaluate_model_predictions must return all required per-horizon keys."""
        res = evaluate_model_predictions(
            self.y_true, self.y_prob,
            model_name="TestModel", threshold=0.5,
        )

        self.assertEqual(res["model_name"], "TestModel")
        self.assertIn("horizons", res)
        self.assertIn("overall_mean", res)

        for step in range(1, self.h + 1):
            h_key = f"horizon_{step}"
            self.assertIn(h_key, res)  # top-level alias
            self.assertIn(h_key, res["horizons"])

            h_m = res[h_key]
            for metric in ["precision", "recall", "f1", "fpr", "pr_auc", "roc_auc", "brier_score"]:
                self.assertIn(metric, h_m)
                self.assertIsInstance(h_m[metric], float)

    def test_evaluate_model_predictions_overall_mean(self):
        """overall_mean must expose mean_f1, mean_brier."""
        res = evaluate_model_predictions(self.y_true, self.y_prob)
        om = res["overall_mean"]
        for key in ["mean_f1", "mean_pr_auc", "mean_fpr", "mean_brier"]:
            self.assertIn(key, om)

    def test_evaluate_state_reconstruction_structure(self):
        """evaluate_state_reconstruction must return MSE, MAE, cosine similarity."""
        states_true = np.random.randn(self.n, self.h, self.d).astype(np.float32)
        states_pred = states_true + 0.1 * np.random.randn(self.n, self.h, self.d).astype(np.float32)

        res = evaluate_state_reconstruction(states_true, states_pred)

        self.assertIn("overall_mse", res)
        self.assertIn("overall_mae", res)
        self.assertIn("per_horizon", res)

        for step in range(1, self.h + 1):
            h_key = f"horizon_{step}"
            self.assertIn(h_key, res["per_horizon"])
            h_r = res["per_horizon"][h_key]
            self.assertIn("mse", h_r)
            self.assertIn("mae", h_r)
            self.assertIn("cosine_similarity", h_r)
            self.assertGreater(h_r["cosine_similarity"], 0.8)

    def test_all_benign_evaluation_edge_case(self):
        """Zero-positive test slice must set precision_undefined=True, precision=0."""
        y_zeros = np.zeros((self.n, self.h), dtype=np.float32)
        y_prob  = np.zeros((self.n, self.h), dtype=np.float32)

        res = evaluate_model_predictions(y_zeros, y_prob, threshold=0.5)
        for step in range(1, self.h + 1):
            h_m = res[f"horizon_{step}"]
            self.assertTrue(h_m["precision_undefined"])
            self.assertEqual(h_m["precision"], 0.0)
            self.assertEqual(h_m["recall"], 0.0)

    def test_compare_all_models_structure(self):
        """compare_all_models must return a results dict and a summary DataFrame."""
        import pandas as pd

        wm_probs = np.random.uniform(0, 1, (self.n, self.h)).astype(np.float32)
        lr_probs = np.random.uniform(0, 1, (self.n, self.h)).astype(np.float32)
        rf_probs = np.random.uniform(0, 1, (self.n, self.h)).astype(np.float32)

        all_res, df = compare_all_models(wm_probs, lr_probs, rf_probs, self.y_true)

        self.assertIsInstance(all_res, dict)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("world_model", all_res)
        self.assertIn("logistic_regression", all_res)
        self.assertIn("random_forest", all_res)
        self.assertFalse(df.empty)


if __name__ == "__main__":
    unittest.main()
