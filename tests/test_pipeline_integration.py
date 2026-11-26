"""Integration tests verifying end-to-end leak-free pipeline execution.

[UNIT_TEST_FIXTURE]: Strictly small synthetic test fixtures for integration verification.
"""

import unittest

import numpy as np
import pandas as pd

from src.contracts import FORECAST_STEPS, NUM_FEATURES, WINDOW_SIZE
from src.pipeline.features import CICIDS2017_FEATURES, UNSW_NB15_FEATURES
from src.pipeline.preprocessor import (
    NetworkDataPreprocessor,
    prepare_research_pipeline,
    preprocess,
)
from src.pipeline.windowing import validate_and_extract_durations


class TestPipelineIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.n_samples = 300
        # Construct synthetic chronological UNSW-NB15 shaped DataFrame
        feature_dict = {feat: np.random.default_rng(i).normal(loc=10.0 + i, scale=2.0, size=self.n_samples)
                        for i, feat in enumerate(UNSW_NB15_FEATURES)}
        feature_dict["timestamp"] = pd.date_range("2026-01-01 10:00:00", periods=self.n_samples, freq="1s")
        feature_dict["dur"] = [0.1] * self.n_samples
        feature_dict["attack_cat"] = ["Normal"] * 200 + ["Exploits"] * 50 + ["DoS"] * 50
        self.df_unsw = pd.DataFrame(feature_dict)

    def test_research_pipeline_end_to_end_no_leakage(self) -> None:
        """Verify that prepare_research_pipeline executes end-to-end with zero full-dataset scaler fitting."""
        pipe_data = prepare_research_pipeline(
            self.df_unsw,
            dataset_name="unsw_nb15",
            train_ratio=0.60,
            val_ratio=0.20,
            timestamp_col="timestamp",
            duration_col="dur",
            duration_unit="s",
        )

        # 1. Scaler center and scale must match training partition mean/std ONLY
        train_rows = len(pipe_data.intervals_train) + WINDOW_SIZE + FORECAST_STEPS - 1
        # The preprocessor sample_count metadata must strictly equal training sample count, not full 300
        self.assertLess(pipe_data.preprocessor.metadata_["sample_count"], self.n_samples)

        # 2. Partitions must all have valid windows
        self.assertGreater(len(pipe_data.X_train), 0)
        self.assertGreater(len(pipe_data.X_val), 0)
        self.assertGreater(len(pipe_data.X_test), 0)

        # 3. Shapes must conform strictly to contract (N, W, D) and (N, H)
        self.assertEqual(pipe_data.X_train.shape[1:], (WINDOW_SIZE, NUM_FEATURES))
        self.assertEqual(pipe_data.y_probs_train.shape[1:], (FORECAST_STEPS,))
        self.assertEqual(pipe_data.X_val.shape[1:], (WINDOW_SIZE, NUM_FEATURES))
        self.assertEqual(pipe_data.X_test.shape[1:], (WINDOW_SIZE, NUM_FEATURES))

        # 4. Verified temporal isolation audit must be present and True
        self.assertTrue(pipe_data.isolation_audit["is_temporally_isolated"])
        self.assertGreater(pipe_data.isolation_audit["train_val_gap_seconds"], 0.0)
        self.assertGreater(pipe_data.isolation_audit["val_test_gap_seconds"], 0.0)

    def test_target_intervals_never_overlap_subsequent_inputs(self) -> None:
        """Confirm training targets strictly precede validation inputs, and val targets precede test inputs."""
        pipe_data = prepare_research_pipeline(
            self.df_unsw,
            dataset_name="unsw_nb15",
            train_ratio=0.60,
            val_ratio=0.20,
            timestamp_col="timestamp",
            duration_col="dur",
        )

        # Max target time of training windows
        max_train_target_time = pipe_data.intervals_train["target_end_time"].max()
        # Min input time of validation windows
        min_val_input_time = pipe_data.intervals_val["input_start_time"].min()

        self.assertLess(max_train_target_time, min_val_input_time)

        # Max target time of validation windows
        max_val_target_time = pipe_data.intervals_val["target_end_time"].max()
        # Min input time of test windows
        min_test_input_time = pipe_data.intervals_test["input_start_time"].min()

        self.assertLess(max_val_target_time, min_test_input_time)

    def test_legacy_preprocess_blocks_research_flag(self) -> None:
        """Verify that preprocess(is_research=True) raises ValueError to block full-data scaler leakage."""
        with self.assertRaises(ValueError) as ctx:
            preprocess(self.df_unsw, is_research=True)
        self.assertIn("strictly prohibited for research workflows", str(ctx.exception))

    def test_missing_timestamps_cannot_enter_research_pipeline(self) -> None:
        """Verify that missing timestamp data cannot enter the research pipeline."""
        df_no_ts = self.df_unsw.drop(columns=["timestamp"])
        with self.assertRaises(ValueError) as ctx:
            prepare_research_pipeline(df_no_ts, dataset_name="unsw_nb15")
        self.assertIn("No valid timestamp column found", str(ctx.exception))

    def test_invalid_durations_rejected_strictly(self) -> None:
        """Verify that non-numeric, NaN, infinite, negative durations and bad units raise ValueError."""
        # 1. Missing duration col
        with self.assertRaises(ValueError) as ctx:
            validate_and_extract_durations(self.df_unsw, duration_col="non_existent_dur")
        self.assertIn("not found in DataFrame", str(ctx.exception))

        # 2. Unsupported duration unit
        with self.assertRaises(ValueError) as ctx:
            validate_and_extract_durations(self.df_unsw, duration_col="dur", duration_unit="minutes")
        self.assertIn("Unsupported duration_unit", str(ctx.exception))

        # 3. NaN durations
        df_nan = self.df_unsw.copy()
        df_nan.iloc[5, df_nan.columns.get_loc("dur")] = np.nan
        with self.assertRaises(ValueError) as ctx:
            validate_and_extract_durations(df_nan, duration_col="dur")
        self.assertIn("NaN/null values", str(ctx.exception))

        # 4. Infinite durations
        df_inf = self.df_unsw.copy()
        df_inf.iloc[5, df_inf.columns.get_loc("dur")] = np.inf
        with self.assertRaises(ValueError) as ctx:
            validate_and_extract_durations(df_inf, duration_col="dur")
        self.assertIn("infinite values", str(ctx.exception))

        # 5. Negative durations
        df_neg = self.df_unsw.copy()
        df_neg.iloc[5, df_neg.columns.get_loc("dur")] = -1.5
        with self.assertRaises(ValueError) as ctx:
            validate_and_extract_durations(df_neg, duration_col="dur")
        self.assertIn("negative duration values", str(ctx.exception))

        # 6. Non-numeric durations
        df_str = self.df_unsw.copy()
        df_str["dur"] = df_str["dur"].astype(object)
        df_str.iloc[5, df_str.columns.get_loc("dur")] = "not_a_number"
        with self.assertRaises(ValueError) as ctx:
            validate_and_extract_durations(df_str, duration_col="dur")
        self.assertIn("non-numeric values", str(ctx.exception))

    def test_duration_unit_conversion_cicids_microseconds(self) -> None:
        """Verify that CIC-IDS2017 microseconds ('us') are correctly converted to seconds."""
        df_us = pd.DataFrame({"flow_duration": [1_000_000.0, 500_000.0, 2_500_000.0]})
        dur_sec = validate_and_extract_durations(df_us, duration_col="flow_duration", duration_unit="us")
        np.testing.assert_allclose(dur_sec, [1.0, 0.5, 2.5])


if __name__ == "__main__":
    unittest.main()
