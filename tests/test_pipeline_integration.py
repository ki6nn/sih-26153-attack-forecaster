"""Integration tests verifying end-to-end leak-free pipeline execution.

[UNIT_TEST_FIXTURE]: Strictly small synthetic test fixtures for integration verification.
"""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.contracts import FORECAST_STEPS, NUM_FEATURES, WINDOW_SIZE
from src.pipeline.features import CICIDS2017_FEATURES, UNSW_NB15_FEATURES
from src.pipeline.ingestion import DatasetProvenance
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
        feature_dict = {
            feat: np.random.default_rng(i).normal(loc=10.0 + i, scale=2.0, size=self.n_samples)
            for i, feat in enumerate(UNSW_NB15_FEATURES)
        }
        feature_dict["timestamp"] = pd.date_range("2026-01-01 10:00:00", periods=self.n_samples, freq="1s")
        feature_dict["dur"] = [0.1] * self.n_samples
        feature_dict["attack_cat"] = ["Normal"] * 200 + ["Exploits"] * 50 + ["DoS"] * 50
        self.df_unsw = pd.DataFrame(feature_dict)

        # Official mock provenance record for unit testing research pipeline
        self.mock_prov = DatasetProvenance(
            dataset_name="unsw_nb15",
            provenance_type="official_research",
            source_url="https://research.unsw.edu.au/projects/unsw-nb15-dataset",
            dataset_version="UNSW-NB15 Mock Research Fixture",
            filename="mock_unsw.csv",
            is_research_evidence=True,
        )
        self.df_unsw.attrs["provenance"] = self.mock_prov

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

        # 1. Preprocessor sample_count metadata must strictly equal training sample count, not full 300
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
            duration_unit="s",
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
        df_no_ts = self.df_unsw.copy()
        df_no_ts = df_no_ts.drop(columns=["timestamp"])
        df_no_ts.attrs["provenance"] = self.mock_prov
        with self.assertRaises(ValueError) as ctx:
            prepare_research_pipeline(df_no_ts, dataset_name="unsw_nb15")
        self.assertIn("no valid timestamp column found", str(ctx.exception).lower())

    def test_strict_provenance_enforcement_rejected_without_provenance(self) -> None:
        """DataFrames without verified provenance must be rejected in research mode."""
        df_unverified = self.df_unsw.copy()
        df_unverified.attrs.clear()
        with self.assertRaises(ValueError) as ctx:
            prepare_research_pipeline(df_unverified, dataset_name="unsw_nb15", enforce_provenance=True)
        self.assertIn("requires explicit dataset provenance", str(ctx.exception))

    def test_synthetic_demo_cannot_enter_research_pipeline(self) -> None:
        """sample_demo.csv must be strictly blocked from entering research evaluation."""
        demo_path = Path("data/sample_demo.csv")
        if demo_path.exists():
            with self.assertRaises(ValueError) as ctx:
                prepare_research_pipeline(demo_path, dataset_name="unsw_nb15", enforce_provenance=True)
            self.assertIn("strictly rejected for empirical research", str(ctx.exception))

    def test_missing_target_labels_rejected(self) -> None:
        """Research pipeline must fail closed when target label columns are absent."""
        df_no_labels = self.df_unsw.drop(columns=["attack_cat"])
        df_no_labels.attrs["provenance"] = self.mock_prov
        with self.assertRaises(ValueError) as ctx:
            prepare_research_pipeline(df_no_labels, dataset_name="unsw_nb15")
        self.assertIn("target label column missing", str(ctx.exception).lower())

    def test_null_target_labels_rejected(self) -> None:
        """Target labels containing null/NaN values must fail loudly."""
        df_null_label = self.df_unsw.copy()
        df_null_label.iloc[10, df_null_label.columns.get_loc("attack_cat")] = np.nan
        df_null_label.attrs["provenance"] = self.mock_prov
        with self.assertRaises(ValueError) as ctx:
            prepare_research_pipeline(df_null_label, dataset_name="unsw_nb15")
        self.assertIn("null/empty/nan values", str(ctx.exception).lower())

    def test_invalid_binary_label_rejected(self) -> None:
        """Non-binary or negative values in 'label' column must be rejected."""
        df_bad_label = self.df_unsw.copy()
        df_bad_label = df_bad_label.drop(columns=["attack_cat"])
        df_bad_label["label"] = [0] * 299 + [-1]  # Invalid -1
        df_bad_label.attrs["provenance"] = self.mock_prov
        with self.assertRaises(ValueError) as ctx:
            prepare_research_pipeline(df_bad_label, dataset_name="unsw_nb15")
        self.assertIn("invalid binary target label values", str(ctx.exception).lower())

    def test_unknown_attack_category_rejected_in_strict_mode(self) -> None:
        """Unrecognized attack categories in attack_cat must raise ValueError when allow_unknown_labels=False."""
        df_unknown_cat = self.df_unsw.copy()
        df_unknown_cat.iloc[10, df_unknown_cat.columns.get_loc("attack_cat")] = "unregistered_zero_day"
        df_unknown_cat.attrs["provenance"] = self.mock_prov
        with self.assertRaises(ValueError) as ctx:
            prepare_research_pipeline(df_unknown_cat, dataset_name="unsw_nb15", allow_unknown_labels=False)
        self.assertIn("unknown attack category labels detected", str(ctx.exception).lower())

    def test_missing_duration_without_acknowledgment_raises(self) -> None:
        """Omitting duration column without explicit zero-duration acknowledgment must raise ValueError."""
        custom_features = [f"feat_{i}" for i in range(NUM_FEATURES)]
        data = {f: [1.0] * 50 for f in custom_features}
        data["timestamp"] = pd.date_range("2026-01-01 10:00:00", periods=50, freq="1s")
        data["attack_cat"] = ["Normal"] * 50
        df_no_dur = pd.DataFrame(data)
        df_no_dur.attrs["provenance"] = self.mock_prov
        with self.assertRaises(ValueError) as ctx:
            prepare_research_pipeline(
                df_no_dur,
                dataset_name="custom",
                feature_columns=custom_features,
                duration_col=None,
                acknowledge_zero_duration=False,
            )
        self.assertIn("flows cannot be assumed instantaneous", str(ctx.exception).lower())

    def test_training_only_imputation_in_research_pipeline(self) -> None:
        """Validation records with NaNs must be imputed using training medians and processed cleanly."""
        df_with_nan = self.df_unsw.copy()
        # Inject NaNs strictly into the validation region (rows 200..230)
        df_with_nan.iloc[210, df_with_nan.columns.get_loc("dur")] = 0.1
        df_with_nan.iloc[210, df_with_nan.columns.get_loc("spkts")] = np.nan
        df_with_nan.attrs["provenance"] = self.mock_prov

        pipe_data = prepare_research_pipeline(
            df_with_nan,
            dataset_name="unsw_nb15",
            train_ratio=0.60,
            val_ratio=0.20,
            timestamp_col="timestamp",
            duration_col="dur",
            duration_unit="s",
        )
        self.assertFalse(np.isnan(pipe_data.X_val).any())
        self.assertEqual(len(pipe_data.X_val), len(pipe_data.intervals_val))

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
