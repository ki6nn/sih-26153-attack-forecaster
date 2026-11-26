"""Tests for chronological data splitting, boundary buffers, duration isolation, and rolling window generation.

[UNIT_TEST_FIXTURE]: Strictly small synthetic test fixtures for windowing verification, never research evidence.
"""

import unittest

import numpy as np
import pandas as pd

from src.contracts import FORECAST_STEPS, NUM_FEATURES, WINDOW_SIZE
from src.pipeline.windowing import (
    chronological_split,
    create_network_windows,
    verify_temporal_isolation,
)


class TestWindowing(unittest.TestCase):
    def setUp(self) -> None:
        self.n_samples = 300
        # Ordered synthetic sequence with 1-second timestamps
        self.df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01 10:00:00", periods=self.n_samples, freq="1s"),
                "value": np.arange(self.n_samples),
                "dur": [0.1] * self.n_samples,
                "attack": (np.arange(self.n_samples) > 200).astype(int),
                "stage": np.where(np.arange(self.n_samples) > 200, 4, 0),
            }
        )

    def test_chronological_split_ordering_and_isolation(self) -> None:
        """Verify strict chronological monotonicity and active isolation verification across partitions."""
        train_df, val_df, test_df = chronological_split(
            self.df,
            train_ratio=0.70,
            val_ratio=0.15,
            timestamp_col="timestamp",
            duration_col="dur",
        )

        # 1. Monotonicity within each partition
        self.assertTrue(train_df["timestamp"].is_monotonic_increasing)
        self.assertTrue(val_df["timestamp"].is_monotonic_increasing)
        self.assertTrue(test_df["timestamp"].is_monotonic_increasing)

        # 2. Strict precedence between partitions
        train_end_time = train_df["timestamp"].max() + pd.to_timedelta(train_df["dur"].max(), unit="s")
        val_start_time = val_df["timestamp"].min()
        val_end_time = val_df["timestamp"].max() + pd.to_timedelta(val_df["dur"].max(), unit="s")
        test_start_time = test_df["timestamp"].min()

        self.assertLess(train_end_time, val_start_time)
        self.assertLess(val_end_time, test_start_time)

    def test_duplicate_timestamp_groups_at_both_boundaries(self) -> None:
        """Verify that split points with duplicate timestamps do not divide the timestamp cluster."""
        # 150 rows total: clusters of 50 identical timestamps
        df_dups = pd.DataFrame(
            {
                "timestamp": [
                    pd.Timestamp("2026-01-01 10:00:00")
                ] * 60
                + [
                    pd.Timestamp("2026-01-01 10:00:10")
                ] * 40
                + [
                    pd.Timestamp("2026-01-01 10:00:20")
                ] * 50,
                "value": np.arange(150),
                "dur": [0.0] * 150,
            }
        )
        train_df, val_df, test_df = chronological_split(
            df_dups,
            train_ratio=0.40,
            val_ratio=0.30,
            timestamp_col="timestamp",
        )
        # Train and val must not share the same timestamp
        self.assertLess(train_df["timestamp"].max(), val_df["timestamp"].min())
        self.assertLess(val_df["timestamp"].max(), test_df["timestamp"].min())
        # Sets of timestamps must be strictly disjoint
        self.assertEqual(len(set(train_df["timestamp"]).intersection(set(val_df["timestamp"]))), 0)
        self.assertEqual(len(set(val_df["timestamp"]).intersection(set(test_df["timestamp"]))), 0)

    def test_unsorted_timestamps_fail_loudly(self) -> None:
        """Verify that an unsorted DataFrame raises ValueError."""
        df_unsorted = self.df.copy()
        # Swap rows to disrupt chronological order
        df_unsorted.iloc[0, df_unsorted.columns.get_loc("timestamp")] = pd.Timestamp("2026-01-02")
        with self.assertRaises(ValueError) as ctx:
            chronological_split(df_unsorted, timestamp_col="timestamp")
        self.assertIn("not sorted chronologically", str(ctx.exception))

    def test_null_timestamps_fail_loudly(self) -> None:
        """Verify that null/NaT timestamps raise ValueError."""
        df_nulls = self.df.copy()
        df_nulls.iloc[10, df_nulls.columns.get_loc("timestamp")] = pd.NaT
        with self.assertRaises(ValueError) as ctx:
            chronological_split(df_nulls, timestamp_col="timestamp")
        self.assertIn("null/nat", str(ctx.exception).lower())

    def test_missing_timestamp_column_fails_loudly_without_fallback(self) -> None:
        """Verify that omitting timestamps raises ValueError unless allow_index_fallback is set."""
        df_no_ts = self.df.drop(columns=["timestamp"])
        with self.assertRaises(ValueError) as ctx:
            chronological_split(df_no_ts, allow_index_fallback=False)
        self.assertIn("No valid timestamp column found", str(ctx.exception))

    def test_long_duration_flows_crossing_boundaries(self) -> None:
        """Verify that a long-duration flow in training pushes validation start past its end time."""
        # 250 rows total so that validation and test partitions have plenty of room
        df_long = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01 10:00:00", periods=250, freq="1s"),
                "value": np.arange(250),
                "dur": [1.0] * 250,
            }
        )
        # Introduce a 45-second flow near the training boundary (row 100)
        df_long.iloc[100, df_long.columns.get_loc("dur")] = 45.0

        train_df, val_df, _ = chronological_split(
            df_long,
            train_ratio=0.50,
            val_ratio=0.25,
            timestamp_col="timestamp",
            duration_col="dur",
        )

        train_max_completion = (train_df["timestamp"] + pd.to_timedelta(train_df["dur"], unit="s")).max()
        val_start_time = val_df["timestamp"].min()

        # Validation must start strictly AFTER the 45-second flow finished
        self.assertGreater(val_start_time, train_max_completion)

    def test_empty_or_insufficient_partitions_raise(self) -> None:
        """Verify that splitting a too-short dataset raises ValueError."""
        df_short = self.df.iloc[:20].copy()
        with self.assertRaises(ValueError):
            chronological_split(df_short, timestamp_col="timestamp")

    def test_verify_temporal_isolation_active_pipeline_use(self) -> None:
        """Verify that verify_temporal_isolation actively blocks temporal overlap."""
        train = pd.DataFrame({"datetime": [pd.Timestamp("2026-01-01 10:00:00")]})
        val = pd.DataFrame({"datetime": [pd.Timestamp("2026-01-01 10:00:00")]})  # Same time!
        test = pd.DataFrame({"datetime": [pd.Timestamp("2026-01-01 10:00:02")]})

        with self.assertRaises(ValueError) as ctx:
            verify_temporal_isolation(train, val, test, timestamp_col="datetime")
        self.assertIn("Temporal Leakage Detected", str(ctx.exception))

    def test_create_network_windows_dimensions_and_intervals(self) -> None:
        """Verify sliding window construction and explicit interval audit metadata."""
        N = 35
        X = np.ones((N, NUM_FEATURES), dtype=np.float32)
        y_attack = (np.arange(N) > 20).astype(int)
        timestamps = pd.date_range("2026-01-01 10:00:00", periods=N, freq="1s")

        X_win, y_fut, y_stg, interval_df = create_network_windows(
            X,
            y_attack,
            timestamps=timestamps,
            window_size=WINDOW_SIZE,
            forecast_steps=FORECAST_STEPS,
            return_intervals=True,
        )

        expected_m = N - WINDOW_SIZE - FORECAST_STEPS + 1
        self.assertEqual(X_win.shape, (expected_m, WINDOW_SIZE, NUM_FEATURES))
        self.assertEqual(y_fut.shape, (expected_m, FORECAST_STEPS))
        self.assertEqual(len(interval_df), expected_m)

        # Check that target start time is strictly after input end time for all windows
        for _, row in interval_df.iterrows():
            self.assertLess(row["input_end_time"], row["target_start_time"])
            self.assertEqual(row["target_start_idx"], row["input_end_idx"] + 1)

    def test_single_class_edge_cases(self) -> None:
        """Verify windowing functions smoothly when all samples belong to a single class."""
        X = np.ones((30, NUM_FEATURES), dtype=np.float32)
        y_all_benign = np.zeros(30, dtype=int)
        _, y_fut, y_stg = create_network_windows(X, y_all_benign)
        self.assertTrue((y_fut == 0.0).all())
        self.assertTrue((y_stg == 0).all())

        y_all_attack = np.ones(30, dtype=int)
        _, y_fut, y_stg = create_network_windows(X, y_all_attack)
        self.assertTrue((y_fut == 1.0).all())
        self.assertTrue((y_stg == 1).all())


if __name__ == "__main__":
    unittest.main()
