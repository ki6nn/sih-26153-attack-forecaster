"""Tests for chronological data splitting, boundary buffers, and rolling window generation.

[UNIT_TEST_FIXTURE]: Strictly small synthetic test fixtures for windowing verification.
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
        # Create ordered synthetic sequence
        self.df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=self.n_samples, freq="1s"),
                "value": np.arange(self.n_samples),
                "attack": (np.arange(self.n_samples) > 200).astype(int),
                "stage": np.where(np.arange(self.n_samples) > 200, 4, 0),
            }
        )

    def test_chronological_split_ordering_and_buffers(self) -> None:
        buffer_size = 13
        train_df, val_df, test_df = chronological_split(
            self.df,
            train_ratio=0.70,
            val_ratio=0.15,
            buffer_size=buffer_size,
        )

        # 1. Verify chronological monotonicity within each split
        self.assertTrue(train_df["timestamp"].is_monotonic_increasing)
        self.assertTrue(val_df["timestamp"].is_monotonic_increasing)
        self.assertTrue(test_df["timestamp"].is_monotonic_increasing)

        # 2. Verify strict chronological ordering across partitions
        self.assertLess(train_df["timestamp"].max(), val_df["timestamp"].min())
        self.assertLess(val_df["timestamp"].max(), test_df["timestamp"].min())

        # 3. Verify boundary buffer exclusion (no overlapping or adjacent records)
        train_max_idx = train_df["value"].max()
        val_min_idx = val_df["value"].min()
        val_max_idx = val_df["value"].max()
        test_min_idx = test_df["value"].min()

        self.assertGreaterEqual(val_min_idx - train_max_idx, buffer_size)
        self.assertGreaterEqual(test_min_idx - val_max_idx, buffer_size)

    def test_chronological_split_duplicate_timestamp_guard(self) -> None:
        """Verify that split points with duplicate timestamps do not split the cluster across partitions."""
        df_dups = pd.DataFrame(
            {
                # Create duplicate timestamps around index 70
                "timestamp": [10.0] * 70 + [20.0] * 20 + [30.0] * 30,
                "value": np.arange(120),
            }
        )
        train_df, val_df, test_df = chronological_split(
            df_dups,
            train_ratio=0.50,
            val_ratio=0.25,
            timestamp_col="timestamp",
            buffer_size=5,
        )
        # Train and val must not share the same timestamp
        self.assertLess(train_df["timestamp"].max(), val_df["timestamp"].min())
        self.assertLess(val_df["timestamp"].max(), test_df["timestamp"].min())

    def test_create_network_windows_dimensions(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.normal(size=(50, NUM_FEATURES)).astype(np.float32)
        y_attack = (rng.uniform(size=50) > 0.5).astype(int)
        y_stage = y_attack * 2

        X_win, y_fut, y_stg = create_network_windows(
            X,
            y_attack,
            y_stage=y_stage,
            window_size=WINDOW_SIZE,
            forecast_steps=FORECAST_STEPS,
        )

        expected_m = 50 - WINDOW_SIZE - FORECAST_STEPS + 1
        self.assertEqual(X_win.shape, (expected_m, WINDOW_SIZE, NUM_FEATURES))
        self.assertEqual(y_fut.shape, (expected_m, FORECAST_STEPS))
        self.assertEqual(y_stg.shape, (expected_m,))

    def test_window_content_alignment(self) -> None:
        """Verify that windows contain exact historical slices and future targets."""
        N = 25
        X = np.arange(N * NUM_FEATURES).reshape(N, NUM_FEATURES).astype(np.float32)
        y_attack = np.arange(N, dtype=np.float32)
        y_stage = np.arange(N, dtype=np.int64)

        X_win, y_fut, y_stg = create_network_windows(
            X, y_attack, y_stage, window_size=WINDOW_SIZE, forecast_steps=FORECAST_STEPS
        )

        # Window 0 should contain X[0..9]
        np.testing.assert_array_equal(X_win[0], X[:WINDOW_SIZE])
        # Future targets for window 0 should contain y_attack[10..13]
        np.testing.assert_array_equal(y_fut[0], y_attack[WINDOW_SIZE : WINDOW_SIZE + FORECAST_STEPS])
        # Next stage should be y_stage[10]
        self.assertEqual(y_stg[0], y_stage[WINDOW_SIZE])

    def test_create_network_windows_with_intervals(self) -> None:
        """Verify explicit interval audit tracking for each constructed window."""
        N = 30
        X = np.ones((N, NUM_FEATURES), dtype=np.float32)
        y_attack = np.zeros(N, dtype=int)
        timestamps = pd.date_range("2026-01-01 10:00:00", periods=N, freq="1s")

        _, _, _, interval_df = create_network_windows(
            X,
            y_attack,
            timestamps=timestamps,
            window_size=WINDOW_SIZE,
            forecast_steps=FORECAST_STEPS,
            return_intervals=True,
        )

        self.assertEqual(len(interval_df), N - WINDOW_SIZE - FORECAST_STEPS + 1)
        # Check window 0 interval
        w0 = interval_df.iloc[0]
        self.assertEqual(w0["input_start_idx"], 0)
        self.assertEqual(w0["input_end_idx"], WINDOW_SIZE - 1)
        self.assertEqual(w0["target_start_idx"], WINDOW_SIZE)
        self.assertEqual(w0["target_end_idx"], WINDOW_SIZE + FORECAST_STEPS - 1)
        self.assertLess(w0["input_end_time"], w0["target_start_time"])

    def test_verify_temporal_isolation_pass_and_fail(self) -> None:
        # Strict precedence should pass
        self.assertTrue(
            verify_temporal_isolation(
                train_targets_end=pd.Timestamp("2026-01-01 10:00:00"),
                val_inputs_start=pd.Timestamp("2026-01-01 10:00:01"),
                val_targets_end=pd.Timestamp("2026-01-01 11:00:00"),
                test_inputs_start=pd.Timestamp("2026-01-01 11:00:01"),
            )
        )

        # Overlapping boundary must raise ValueError
        with self.assertRaises(ValueError):
            verify_temporal_isolation(
                train_targets_end=pd.Timestamp("2026-01-01 10:00:05"),
                val_inputs_start=pd.Timestamp("2026-01-01 10:00:02"),  # Leakage!
                val_targets_end=pd.Timestamp("2026-01-01 11:00:00"),
                test_inputs_start=pd.Timestamp("2026-01-01 11:00:01"),
            )

    def test_insufficient_samples_raises(self) -> None:
        X_short = np.zeros((10, NUM_FEATURES))
        y_short = np.zeros(10)
        with self.assertRaises(ValueError):
            create_network_windows(X_short, y_short, window_size=10, forecast_steps=4)

    def test_single_class_edge_cases(self) -> None:
        """Verify windowing functions smoothly when all samples belong to a single class."""
        X = np.ones((30, NUM_FEATURES), dtype=np.float32)
        # 100% Benign
        y_all_benign = np.zeros(30, dtype=int)
        _, y_fut, y_stg = create_network_windows(X, y_all_benign)
        self.assertTrue((y_fut == 0.0).all())
        self.assertTrue((y_stg == 0).all())

        # 100% Attack
        y_all_attack = np.ones(30, dtype=int)
        _, y_fut, y_stg = create_network_windows(X, y_all_attack)
        self.assertTrue((y_fut == 1.0).all())
        self.assertTrue((y_stg == 1).all())


if __name__ == "__main__":
    unittest.main()
