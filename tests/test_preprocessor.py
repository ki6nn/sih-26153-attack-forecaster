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


if __name__ == "__main__":
    unittest.main()
