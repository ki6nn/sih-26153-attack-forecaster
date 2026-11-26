"""Tests for dataset provenance records and checksum verifications.

[UNIT_TEST_FIXTURE]: Strictly tests provenance validation utilities and records.
"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.pipeline.ingestion import (
    DatasetProvenance,
    compute_file_sha256,
    verify_dataset_provenance,
)


class TestProvenance(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]
        self.demo_path = self.root / "data" / "sample_demo.csv"
        self.provenance_path = self.root / "data" / "sample_demo_provenance.json"

    def test_sample_demo_provenance_file_exists(self) -> None:
        self.assertTrue(self.provenance_path.exists(), "Provenance JSON must exist for sample_demo.csv")

    def test_sample_demo_provenance_content_and_checksum(self) -> None:
        with open(self.provenance_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.assertEqual(meta["file"], "data/sample_demo.csv")
        self.assertEqual(meta["provenance_type"], "synthetic_demonstration_data")
        self.assertFalse(meta["is_research_evidence"], "Sample demo must NOT be marked as research evidence")
        self.assertIn("usage_restriction", meta)

        if self.demo_path.exists():
            computed_hash = compute_file_sha256(self.demo_path)
            self.assertEqual(computed_hash, meta["sha256"].lower())

    def test_synthetic_data_rejected_in_research_mode(self) -> None:
        """Sample demo must be strictly rejected when evaluated in research mode."""
        if self.demo_path.exists():
            with self.assertRaises(ValueError) as ctx:
                verify_dataset_provenance(self.demo_path, is_research=True)
            self.assertIn("strictly rejected for empirical research", str(ctx.exception))

    def test_synthetic_data_allowed_in_non_research_mode(self) -> None:
        """Sample demo is permitted for offline demo when is_research=False."""
        if self.demo_path.exists():
            prov = verify_dataset_provenance(self.demo_path, is_research=False)
            self.assertFalse(prov.is_research_evidence)
            self.assertEqual(prov.provenance_type, "synthetic_demonstration_data")

    def test_provenance_valid_official_verification(self) -> None:
        """Verify that an official dataset with matching checksum passes research verification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_unsw.csv"
            csv_path.write_text("dur,spkts,label\n0.1,10,0\n", encoding="utf-8")
            file_hash = compute_file_sha256(csv_path)

            prov = DatasetProvenance(
                dataset_name="unsw_nb15",
                provenance_type="official_research",
                source_url="https://research.unsw.edu.au/projects/unsw-nb15-dataset",
                dataset_version="UNSW-NB15 Raw Release",
                filename=csv_path.name,
                sha256=file_hash,
                is_research_evidence=True,
            )
            verified = verify_dataset_provenance(csv_path, provenance=prov, is_research=True)
            self.assertTrue(verified.is_research_evidence)
            self.assertEqual(verified.dataset_name, "unsw_nb15")

    def test_provenance_checksum_mismatch_raises(self) -> None:
        """Verify that modified content with an incorrect checksum raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "tampered.csv"
            csv_path.write_text("modified content\n", encoding="utf-8")

            prov = DatasetProvenance(
                dataset_name="unsw_nb15",
                provenance_type="official_research",
                source_url="https://research.unsw.edu.au/projects/unsw-nb15-dataset",
                filename=csv_path.name,
                sha256="0000000000000000000000000000000000000000000000000000000000000000",
                is_research_evidence=True,
            )
            with self.assertRaises(ValueError) as ctx:
                verify_dataset_provenance(csv_path, provenance=prov, is_research=True)
            self.assertIn("checksum mismatch", str(ctx.exception).lower())

    def test_provenance_missing_rejected_in_research(self) -> None:
        """Verify that missing provenance raises ValueError when enforce_provenance=True or is_research=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "unknown.csv"
            csv_path.write_text("col1,col2\n1,2\n", encoding="utf-8")

            with self.assertRaises(ValueError) as ctx:
                verify_dataset_provenance(csv_path, is_research=True)
            self.assertIn("could not be verified", str(ctx.exception).lower())

    def test_provenance_unknown_dataset_identity_rejected(self) -> None:
        """Verify that an unrecognized dataset name raises ValueError in research mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "custom.csv"
            csv_path.write_text("col1,col2\n1,2\n", encoding="utf-8")
            file_hash = compute_file_sha256(csv_path)

            prov = DatasetProvenance(
                dataset_name="alien_benchmark_2026",
                provenance_type="official_research",
                source_url="https://example.com/data",
                filename=csv_path.name,
                sha256=file_hash,
                is_research_evidence=True,
            )
            with self.assertRaises(ValueError) as ctx:
                verify_dataset_provenance(csv_path, provenance=prov, is_research=True)
            self.assertIn("not recognized as an approved official research benchmark", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
