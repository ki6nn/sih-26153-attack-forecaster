"""Tests for data validation, missing/infinite value detection, and quality reports.

[UNIT_TEST_FIXTURE]: Strictly small synthetic test fixtures for code testing.
"""

import unittest

import numpy as np
import pandas as pd

from src.pipeline.validation import (
    clean_missing_and_infinite,
    detect_duplicate_records,
    detect_missing_and_infinite,
    generate_data_quality_report,
    validate_labels,
    validate_required_columns,
)


class TestValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_df = pd.DataFrame(
            {
                "feat_a": [1.0, np.nan, 3.0, np.inf, 5.0],
                "feat_b": [10.0, 20.0, -np.inf, 40.0, 50.0],
                "feat_c": [1.0, 2.0, 3.0, 4.0, 5.0],
                "attack_cat": ["Normal", "DoS", "Exploits", "Normal", "Normal"],
            }
        )

    def test_validate_required_columns_success(self) -> None:
        validate_required_columns(self.sample_df, ["feat_a", "feat_b"])

    def test_validate_required_columns_missing_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_required_columns(self.sample_df, ["feat_a", "missing_feat"])
        self.assertIn("missing_feat", str(ctx.exception))

    def test_detect_missing_and_infinite(self) -> None:
        report = detect_missing_and_infinite(self.sample_df, columns=["feat_a", "feat_b"])
        self.assertEqual(report["feat_a"]["nan_count"], 1)
        self.assertEqual(report["feat_a"]["inf_count"], 1)
        self.assertEqual(report["feat_b"]["inf_count"], 1)

    def test_clean_missing_and_infinite_with_median(self) -> None:
        cleaned, fill_vals = clean_missing_and_infinite(self.sample_df, columns=["feat_a", "feat_b"])
        # Check no Infs or NaNs remain
        self.assertFalse(np.isinf(cleaned["feat_a"]).any())
        self.assertFalse(cleaned["feat_a"].isna().any())
        self.assertFalse(np.isinf(cleaned["feat_b"]).any())
        self.assertFalse(cleaned["feat_b"].isna().any())
        self.assertIn("feat_a", fill_vals)

    def test_clean_with_provided_fill_values(self) -> None:
        test_df = pd.DataFrame({"feat_a": [np.nan, np.inf, 10.0]})
        train_fills = {"feat_a": 42.0}
        cleaned, _ = clean_missing_and_infinite(test_df, fill_values=train_fills)
        # NaN and Inf should both be replaced by train_fills
        self.assertEqual(cleaned["feat_a"].iloc[0], 42.0)
        self.assertEqual(cleaned["feat_a"].iloc[1], 42.0)
        self.assertEqual(cleaned["feat_a"].iloc[2], 10.0)

    def test_validate_labels(self) -> None:
        audit = validate_labels(
            self.sample_df,
            label_col="attack_cat",
            valid_labels=["Normal", "DoS", "Exploits"],
        )
        self.assertTrue(audit["is_valid"])
        self.assertEqual(audit["total_records"], 5)
        self.assertEqual(audit["unique_classes"], 3)

    def test_validate_labels_unexpected(self) -> None:
        df = pd.DataFrame({"attack_cat": ["Normal", "UnknownThreat"]})
        audit = validate_labels(df, label_col="attack_cat", valid_labels=["Normal"])
        self.assertFalse(audit["is_valid"])
        self.assertIn("UnknownThreat", audit["unexpected_classes"])

    def test_detect_duplicates(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 1, 3], "b": [10, 20, 10, 30]})
        count, mask = detect_duplicate_records(df)
        self.assertEqual(count, 1)
        self.assertTrue(mask.iloc[2])

    def test_generate_data_quality_report(self) -> None:
        report = generate_data_quality_report(
            self.sample_df,
            dataset_name="unit_test_fixture",
            numeric_columns=["feat_a", "feat_b", "feat_c"],
            label_col="attack_cat",
        )
        self.assertEqual(report["total_rows"], 5)
        self.assertIn("feat_a", report["columns_with_missing_or_inf"])


if __name__ == "__main__":
    unittest.main()
