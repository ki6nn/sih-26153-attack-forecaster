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


if __name__ == "__main__":
    unittest.main()
