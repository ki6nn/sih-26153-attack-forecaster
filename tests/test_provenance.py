"""Tests for dataset provenance records and checksum verifications.

[UNIT_TEST_FIXTURE]: Strictly tests provenance validation utilities and records.
"""

import hashlib
import json
import unittest
from pathlib import Path


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
            with open(self.demo_path, "rb") as f:
                computed_hash = hashlib.sha256(f.read()).hexdigest().lower()
            self.assertEqual(computed_hash, meta["sha256"].lower())


if __name__ == "__main__":
    unittest.main()
