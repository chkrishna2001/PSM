import json
import tempfile
import unittest
from pathlib import Path

from prod_memory.build_prod_extraction_v4_fixture_repair import (
    FAILING_FIXTURE_IDS,
    PROD_EXTRACTION_V4_PROFILE,
    build_prod_extraction_v4_fixture_repair,
)
from prod_memory.row_validation import validate_prod_row

REPO_ROOT = Path(__file__).resolve().parents[3]
DIRECT_PROBES = REPO_ROOT / "psm-model" / "data" / "probes" / "direct_probes.jsonl"


class BuildProdExtractionV4Test(unittest.TestCase):
    def test_fixture_repair_mix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "prod-extraction-v4.jsonl"
            profile = {
                **PROD_EXTRACTION_V4_PROFILE,
                "fail_copies": 2,
                "pass_copies": 1,
                "noise_copies": 1,
                "expanded_copies": 1,
                "recall_copies": 1,
            }
            manifest = build_prod_extraction_v4_fixture_repair(
                output,
                direct_probes=DIRECT_PROBES if DIRECT_PROBES.exists() else None,
                profile=profile,
            )
            self.assertEqual(manifest["profile"], "prod-extraction-v4-fixture-repair")
            self.assertTrue(manifest["validation"]["ok"])
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertGreaterEqual(len(rows), len(FAILING_FIXTURE_IDS) * profile["fail_copies"])
            fail_prefix = next(row for row in rows if row["id"].startswith("repair-fail:"))
            self.assertIn("fixture-plan-01-handoff", fail_prefix["id"])
            validate_prod_row(fail_prefix)


if __name__ == "__main__":
    unittest.main()
