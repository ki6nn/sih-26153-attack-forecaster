"""Tests for feature extraction, categorical encoding, and label mapping.

[UNIT_TEST_FIXTURE]: Strictly small synthetic test fixtures for feature verification.
"""

import unittest

import numpy as np
import pandas as pd

from src.contracts import NUM_FEATURES
from src.pipeline.features import (
    CICIDS2017_FEATURES,
    UNSW_NB15_FEATURES,
    detect_feature_anomalies,
    encode_categorical_features,
    extract_features,
    map_stage_labels,
)


class TestFeatures(unittest.TestCase):
    def test_feature_list_lengths(self) -> None:
        self.assertEqual(len(CICIDS2017_FEATURES), NUM_FEATURES)
        self.assertEqual(len(UNSW_NB15_FEATURES), NUM_FEATURES)

    def test_extract_features_unsw(self) -> None:
        data = {col: [1.0, 2.0] for col in UNSW_NB15_FEATURES}
        df = pd.DataFrame(data)
        extracted = extract_features(df, dataset_name="unsw_nb15")
        self.assertEqual(extracted.shape, (2, NUM_FEATURES))
        self.assertEqual(extracted.dtypes.iloc[0], np.float32)

    def test_extract_features_missing_raises(self) -> None:
        df = pd.DataFrame({"dur": [1.0], "spkts": [2.0]})
        with self.assertRaises(ValueError) as ctx:
            extract_features(df, dataset_name="unsw_nb15")
        self.assertIn("missing", str(ctx.exception).lower())

    def test_map_stage_labels_unsw(self) -> None:
        raw_labels = ["normal", "Reconnaissance", "exploits", "Backdoors", "dos", "unknown_variant"]
        mapped = map_stage_labels(raw_labels, dataset_name="unsw_nb15")
        expected = ["Normal", "Reconnaissance", "Exploits", "Backdoor", "DoS", "Generic"]
        self.assertEqual(mapped.tolist(), expected)

    def test_map_stage_labels_cicids(self) -> None:
        raw_labels = ["BENIGN", "PortScan", "DDoS", "Web Attack – Brute Force", "Infiltration"]
        mapped = map_stage_labels(raw_labels, dataset_name="cic_ids2017")
        expected = ["Normal", "Reconnaissance", "DoS", "Exploits", "Backdoor"]
        self.assertEqual(mapped.tolist(), expected)

    def test_encode_categorical_features_no_target_leakage(self) -> None:
        train_df = pd.DataFrame({"proto": ["tcp", "udp", "tcp", "icmp"]})
        encoded_train, encoders = encode_categorical_features(train_df, ["proto"])
        self.assertIn("proto", encoders)

        # Test set with unseen category 'arp'
        test_df = pd.DataFrame({"proto": ["tcp", "arp"]})
        encoded_test, _ = encode_categorical_features(test_df, ["proto"], encoders=encoders)
        # 'tcp' must retain the exact training code; 'arp' should receive the unknown code
        tcp_code = encoders["proto"]["tcp"]
        unknown_code = encoders["proto"]["unknown"]
        self.assertEqual(encoded_test["proto"].iloc[0], tcp_code)
        self.assertEqual(encoded_test["proto"].iloc[1], unknown_code)

    def test_detect_feature_anomalies(self) -> None:
        """Detect NaNs, +/- Infs, and non-numeric strings across features."""
        df = pd.DataFrame(
            {
                "feat_num": [1.0, np.nan, np.inf, -np.inf, 5.0],
                "feat_str": ["10", "corrupted", "30", "40", "50"],
                "feat_clean": [0.0, 1.0, 2.0, 3.0, 4.0],
            }
        )
        anomalies = detect_feature_anomalies(df, ["feat_num", "feat_str", "feat_clean"])
        self.assertIn("feat_num", anomalies)
        self.assertEqual(anomalies["feat_num"]["nan_count"], 1)
        self.assertEqual(anomalies["feat_num"]["posinf_count"], 1)
        self.assertEqual(anomalies["feat_num"]["neginf_count"], 1)

        self.assertIn("feat_str", anomalies)
        self.assertEqual(anomalies["feat_str"]["non_numeric_count"], 1)

        self.assertNotIn("feat_clean", anomalies)

    def test_extract_features_preserves_legitimate_zeros(self) -> None:
        """Legitimate zero-valued network features must not be corrupted or overwritten."""
        data = {col: [0.0, 10.0] for col in UNSW_NB15_FEATURES}
        df = pd.DataFrame(data)
        extracted = extract_features(df, dataset_name="unsw_nb15")
        for col in UNSW_NB15_FEATURES:
            self.assertEqual(extracted[col].iloc[0], 0.0)
            self.assertEqual(extracted[col].iloc[1], 10.0)

    def test_extract_features_with_training_impute_values(self) -> None:
        """Invalid feature values in validation/test must be replaced with training-derived values."""
        data = {col: [np.nan, np.inf] for col in UNSW_NB15_FEATURES}
        df = pd.DataFrame(data)
        train_fills = {col: 42.5 for col in UNSW_NB15_FEATURES}
        extracted = extract_features(df, dataset_name="unsw_nb15", impute_values=train_fills)
        for col in UNSW_NB15_FEATURES:
            self.assertEqual(extracted[col].iloc[0], np.float32(42.5))
            self.assertEqual(extracted[col].iloc[1], np.float32(42.5))

    def test_extract_features_raise_on_invalid_without_imputation(self) -> None:
        """extract_features with raise_on_invalid=True raises ValueError when invalid values are present."""
        data = {col: [1.0, np.nan] for col in UNSW_NB15_FEATURES}
        df = pd.DataFrame(data)
        with self.assertRaises(ValueError) as ctx:
            extract_features(df, dataset_name="unsw_nb15", raise_on_invalid=True)
        self.assertIn("Invalid numeric feature values detected", str(ctx.exception))

    def test_map_stage_labels_null_or_empty_raises(self) -> None:
        """Null, empty string, or NaN labels must fail loudly instead of becoming Normal."""
        bad_labels_cases = [
            ["Normal", None],
            ["Normal", np.nan],
            ["Normal", ""],
            ["Normal", "   "],
            ["Normal", "none"],
        ]
        for bad_labels in bad_labels_cases:
            with self.assertRaises(ValueError) as ctx:
                map_stage_labels(bad_labels, dataset_name="unsw_nb15")
            self.assertIn("null/empty/nan values", str(ctx.exception).lower())

    def test_map_stage_labels_unknown_category(self) -> None:
        """Unknown category must map to Generic (attack) in permissive mode or raise in strict mode."""
        labels = ["normal", "alien_zero_day"]
        mapped = map_stage_labels(labels, dataset_name="unsw_nb15", strict=False)
        self.assertEqual(mapped.iloc[1], "Generic")

        with self.assertRaises(ValueError) as ctx:
            map_stage_labels(labels, dataset_name="unsw_nb15", strict=True, allow_unknown=False)
        self.assertIn("Unknown attack category labels", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
