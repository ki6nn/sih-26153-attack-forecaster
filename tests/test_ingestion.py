"""Tests for data ingestion, column normalization, and timestamp parsing.

[UNIT_TEST_FIXTURE]: The data used in this test file consists strictly of small
synthetic unit test fixtures for algorithmic verification, never research evidence.
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.pipeline.ingestion import (
    load_csv_chunks,
    load_dataset,
    normalize_column_names,
    parse_timestamps,
    record_source_provenance,
)


class TestIngestion(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_csv = Path(self.temp_dir.name) / "test_flows.csv"

        # Construct a synthetic unit-test fixture
        df = pd.DataFrame(
            {
                " Destination Port": [80, 443, 22, 8080],
                "Flow Bytes/s": [1024.5, 2048.0, 512.0, 4096.0],
                "Timestamp": [
                    "2017-07-07 10:00:00",
                    "2017-07-07 09:00:00",  # Deliberately out of order
                    "2017-07-07 11:00:00",
                    "invalid-date",
                ],
                "Label": ["BENIGN", "PortScan", "BENIGN", "DDoS"],
            }
        )
        df.to_csv(self.test_csv, index=False)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_normalize_column_names(self) -> None:
        raw_cols = [" Destination Port", "Flow Bytes/s", "Fwd IAT Total", "Attack-Cat!"]
        normalized = normalize_column_names(raw_cols)
        expected = ["destination_port", "flow_bytes_s", "fwd_iat_total", "attack_cat"]
        self.assertEqual(normalized, expected)

    def test_parse_timestamps_chronological_ordering(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp": [
                    "2026-01-01 12:00:00",
                    "2026-01-01 08:00:00",
                    "2026-01-01 10:00:00",
                ],
                "val": [3, 1, 2],
            }
        )
        parsed = parse_timestamps(df, sort_chronological=True)
        self.assertIn("datetime", parsed.columns)
        self.assertTrue(parsed["datetime"].is_monotonic_increasing)
        self.assertEqual(parsed["val"].tolist(), [1, 2, 3])

    def test_parse_timestamps_epoch_numeric(self) -> None:
        # Test UNSW-NB15 epoch timestamps
        df = pd.DataFrame(
            {
                "stime": [1421981100.0, 1421981000.0, 1421981200.0],
                "val": [2, 1, 3],
            }
        )
        parsed = parse_timestamps(df, timestamp_col="stime", sort_chronological=True)
        self.assertIn("datetime", parsed.columns)
        self.assertTrue(parsed["datetime"].is_monotonic_increasing)
        self.assertEqual(parsed["val"].tolist(), [1, 2, 3])

    def test_invalid_timestamps_graceful_handling(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp": ["corrupted_timestamp", "another_bad_val", None],
                "val": [1, 2, 3],
            }
        )
        parsed = parse_timestamps(df, sort_chronological=True)
        self.assertIn("datetime", parsed.columns)
        self.assertTrue(parsed["datetime"].isna().all())

    def test_record_source_provenance(self) -> None:
        df = pd.DataFrame({"a": [1, 2]})
        provenance_df = record_source_provenance(df, "cic_ids2017", self.test_csv)
        self.assertEqual(provenance_df["_source_dataset"].iloc[0], "cic_ids2017")
        self.assertEqual(provenance_df["_source_file"].iloc[0], "test_flows.csv")

    def test_load_csv_chunks(self) -> None:
        chunks = list(load_csv_chunks(self.test_csv, chunksize=2))
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0]), 2)
        self.assertIn("destination_port", chunks[0].columns)

    def test_load_dataset(self) -> None:
        loaded = load_dataset(self.test_csv, dataset_name="test_data")
        self.assertEqual(len(loaded), 4)
        self.assertIn("destination_port", loaded.columns)
        self.assertEqual(loaded["_source_dataset"].iloc[0], "test_data")


if __name__ == "__main__":
    unittest.main()
