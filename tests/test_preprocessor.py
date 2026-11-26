"""Tests for training-only preprocessing, isolation, and serialization.

[UNIT_TEST_FIXTURE]: Strictly small synthetic test fixtures for preprocessor verification.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.contracts import NUM_FEATURES
from src.pipeline.preprocessor import NetworkDataPreprocessor


class TestPreprocessor(unittest.TestCase):
    def setUp(self) -> None:
        self.feature_names = [f"f_{i}" for i in range(NUM_FEATURES)]
        # Training partition: mean 10, std 2
        rng = np.random.default_rng(42)
        self.X_train = rng.normal(loc=10.0, scale=2.0, size=(100, NUM_FEATURES)).astype(np.float32)
        # Test partition with different distribution: mean 50, scale 10
        self.X_test = rng.normal(loc=50.0, scale=10.0, size=(40, NUM_FEATURES)).astype(np.float32)

    def test_training_only_fitting_isolation(self) -> None:
        """Verify that preprocessor fits strictly on training data and remains unaffected by test data."""
        preprocessor = NetworkDataPreprocessor(feature_columns=self.feature_names)
        preprocessor.fit(self.X_train)

        train_mean = preprocessor.center_.copy()
        train_scale = preprocessor.scale_.copy()

        # Transform test data out-of-sample
        scaled_test = preprocessor.transform(self.X_test)

        # Ensure parameters did not mutate after transforming test data
        np.testing.assert_array_equal(preprocessor.center_, train_mean)
        np.testing.assert_array_equal(preprocessor.scale_, train_scale)
        self.assertEqual(scaled_test.shape, (40, NUM_FEATURES))

    def test_deterministic_transform(self) -> None:
        """Verify that repeated transformations produce identical outputs."""
        preprocessor = NetworkDataPreprocessor(feature_columns=self.feature_names)
        preprocessor.fit(self.X_train)

        out1 = preprocessor.transform(self.X_test)
        out2 = preprocessor.transform(self.X_test)
        np.testing.assert_array_equal(out1, out2)

    def test_constant_feature_zero_division_guard(self) -> None:
        """Ensure constant features with zero standard deviation do not produce NaNs or Inf."""
        X_constant = np.ones((50, NUM_FEATURES), dtype=np.float32) * 5.0
        preprocessor = NetworkDataPreprocessor(feature_columns=self.feature_names)
        scaled = preprocessor.fit_transform(X_constant)

        self.assertFalse(np.isnan(scaled).any())
        self.assertFalse(np.isinf(scaled).any())
        # All centered values should be 0.0
        np.testing.assert_allclose(scaled, np.zeros_like(scaled))

    def test_save_and_load_artifact(self) -> None:
        """Verify serialization and deserialization of fitted preprocessor."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "preprocessor.joblib"
            prep = NetworkDataPreprocessor(feature_columns=self.feature_names)
            prep.fit(self.X_train)
            prep.save(save_path)

            loaded = NetworkDataPreprocessor.load(save_path)
            self.assertTrue(loaded.is_fitted)
            np.testing.assert_array_equal(loaded.center_, prep.center_)
            np.testing.assert_array_equal(loaded.scale_, prep.scale_)

            # Test outputs match exactly
            out_orig = prep.transform(self.X_test)
            out_loaded = loaded.transform(self.X_test)
            np.testing.assert_array_equal(out_orig, out_loaded)

    def test_unfitted_transform_raises(self) -> None:
        preprocessor = NetworkDataPreprocessor(feature_columns=self.feature_names)
        with self.assertRaises(RuntimeError):
            preprocessor.transform(self.X_train)

    def test_dimension_mismatch_raises(self) -> None:
        preprocessor = NetworkDataPreprocessor(feature_columns=self.feature_names)
        preprocessor.fit(self.X_train)
        with self.assertRaises(ValueError):
            # Wrong number of features (5 instead of 20)
            preprocessor.transform(np.zeros((10, 5), dtype=np.float32))

    def test_independent_scaler_center_and_scale_derivation(self) -> None:
        """Independently derive expected training-only center/scale without using preprocessor code.

        Proves that:
            1. Preprocessor center and scale strictly match independently computed training statistics.
            2. Extreme shifts in val/test distributions do not alter fitted training statistics.
            3. Transform never calls fit/fit_transform.
            4. Serialized artifact reproduces exact transformations.
        """
        N_train = 120
        N_val = 40
        N_test = 40

        # Construct distinct distributions
        rng = np.random.default_rng(12345)
        raw_train = rng.uniform(low=5.0, high=25.0, size=(N_train, NUM_FEATURES)).astype(np.float64)
        raw_val = rng.uniform(low=100.0, high=200.0, size=(N_val, NUM_FEATURES)).astype(np.float64)
        raw_test = rng.uniform(low=-50.0, high=-10.0, size=(N_test, NUM_FEATURES)).astype(np.float64)

        # Independent manual calculation of expected center and scale on training partition ONLY
        expected_center = np.zeros(NUM_FEATURES, dtype=np.float32)
        expected_scale = np.zeros(NUM_FEATURES, dtype=np.float32)
        for j in range(NUM_FEATURES):
            col_train = raw_train[:, j]
            # Manual sum and variance
            mean_j = sum(col_train) / float(N_train)
            var_j = sum((x - mean_j) ** 2 for x in col_train) / float(N_train)
            std_j = var_j ** 0.5
            expected_center[j] = np.float32(mean_j)
            expected_scale[j] = np.float32(std_j if std_j >= 1e-6 else 1.0)

        # Fit preprocessor on training partition only
        preprocessor = NetworkDataPreprocessor(feature_columns=self.feature_names, method="standard")
        preprocessor.fit(raw_train)

        # Assert exact equivalence to independent calculation
        np.testing.assert_allclose(preprocessor.center_, expected_center, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(preprocessor.scale_, expected_scale, rtol=1e-5, atol=1e-5)
        self.assertEqual(preprocessor.metadata_["sample_count"], N_train)

        # Snapshot parameters before transforming val/test
        center_snapshot = preprocessor.center_.copy()
        scale_snapshot = preprocessor.scale_.copy()

        # Modify validation and test sets substantially with extreme outliers (e.g. 1e9)
        extreme_val = raw_val.copy() + 1e9
        extreme_test = raw_test.copy() - 1e9

        scaled_val = preprocessor.transform(extreme_val)
        scaled_test = preprocessor.transform(extreme_test)

        # Assert preprocessor statistics remained strictly unchanged
        np.testing.assert_array_equal(preprocessor.center_, center_snapshot)
        np.testing.assert_array_equal(preprocessor.scale_, scale_snapshot)
        self.assertEqual(preprocessor.metadata_["sample_count"], N_train)

        # Verify serialization reload consistency
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_file = Path(tmp_dir) / "test_scaler.joblib"
            preprocessor.save(artifact_file)
            reloaded = NetworkDataPreprocessor.load(artifact_file)

            reloaded_val_scaled = reloaded.transform(extreme_val)
            np.testing.assert_array_equal(scaled_val, reloaded_val_scaled)

    def test_training_only_imputation_isolation(self) -> None:
        """Prove that validation and test non-finite values are imputed strictly using training statistics."""
        N_train = 60
        N_val = 30
        rng = np.random.default_rng(999)

        train_data = rng.normal(loc=25.0, scale=1.0, size=(N_train, NUM_FEATURES)).astype(np.float32)
        # Independent manual calculation of training medians
        expected_impute_vals = {self.feature_names[j]: float(np.median(train_data[:, j])) for j in range(NUM_FEATURES)}

        preprocessor = NetworkDataPreprocessor(
            feature_columns=self.feature_names,
            method="standard",
            impute_strategy="median",
        )
        preprocessor.fit(train_data)

        # Fitted impute values must match training medians
        for col, expected_val in expected_impute_vals.items():
            self.assertAlmostEqual(preprocessor.impute_values_[col], expected_val, places=4)

        # Create validation partition filled with NaNs and Infs
        val_with_nans = np.zeros((N_val, NUM_FEATURES), dtype=np.float32)
        val_with_nans[:] = np.nan
        val_with_nans[0, 0] = np.inf
        val_with_nans[1, 1] = -np.inf

        # Transform validation with NaNs; ensure no NaNs or Infs remain in scaled output
        scaled_val = preprocessor.transform(val_with_nans)
        self.assertFalse(np.isnan(scaled_val).any())
        self.assertFalse(np.isinf(scaled_val).any())

        # Impute values must not have been mutated by validation set
        for col, expected_val in expected_impute_vals.items():
            self.assertAlmostEqual(preprocessor.impute_values_[col], expected_val, places=4)


if __name__ == "__main__":
    unittest.main()
