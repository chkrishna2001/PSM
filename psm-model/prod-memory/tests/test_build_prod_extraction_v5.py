import json
import tempfile
import unittest
from pathlib import Path

from prod_memory.build_prod_extraction_v5_suite_micro import (
    FOCUS_SUITE_FIXTURES,
    PROD_EXTRACTION_V5_PROFILE,
    build_prod_extraction_v5_suite_micro,
)
from prod_memory.row_validation import validate_prod_row

REPO_ROOT = Path(__file__).resolve().parents[3]
DIRECT_PROBES = REPO_ROOT / "psm-model" / "data" / "probes" / "direct_probes.jsonl"


class BuildProdExtractionV5Test(unittest.TestCase):
    def test_plan_chunks_micro_mix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "prod-extraction-v5.jsonl"
            profile = {
                **PROD_EXTRACTION_V5_PROFILE,
                "focus_copies": 2,
                "plan_seed_copies": 2,
                "pass_copies": 1,
                "noise_copies": 1,
                "expanded_copies": 1,
                "recall_copies": 2,
            }
            manifest = build_prod_extraction_v5_suite_micro(
                output,
                focus_suite="plan_chunks",
                direct_probes=DIRECT_PROBES if DIRECT_PROBES.exists() else None,
                profile=profile,
            )
            self.assertEqual(manifest["focus_suite"], "plan_chunks")
            self.assertTrue(manifest["validation"]["ok"])
            self.assertGreaterEqual(manifest["anchor_fraction"], 0.5)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
            focus_ids = FOCUS_SUITE_FIXTURES["plan_chunks"]
            focus_rows = [row for row in rows if any(fid in row["id"] for fid in focus_ids)]
            self.assertGreaterEqual(len(focus_rows), len(focus_ids) * profile["focus_copies"])
            validate_prod_row(focus_rows[0])


if __name__ == "__main__":
    unittest.main()
