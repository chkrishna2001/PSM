import json
import tempfile
import unittest
from pathlib import Path

from prod_memory.build_prod_extraction_v1 import PROD_EXTRACTION_V1_PROFILE, build_prod_extraction_v1
from prod_memory.curriculum_sources import build_fixture_rows, build_primary_source_rows
from prod_memory.row_validation import validate_prod_row, validate_prod_rows

REPO_ROOT = Path(__file__).resolve().parents[3]
DIRECT_PROBES = REPO_ROOT / "psm-model" / "data" / "probes" / "direct_probes.jsonl"


class BuildProdExtractionV1Test(unittest.TestCase):
    def test_primary_rows_validate(self) -> None:
        rows = build_primary_source_rows()
        self.assertGreaterEqual(len(rows), 10)
        report = validate_prod_rows(rows)
        self.assertTrue(report["ok"], report["failures"])

    def test_fixture_rows_include_workflow_indexables(self) -> None:
        rows = build_fixture_rows()
        review = next(row for row in rows if row["id"] == "fixture-workflow-review-pr")
        indexables = review["expected"]["indexables"]
        self.assertTrue(any(item.get("kind") == "workflow" for item in indexables))
        self.assertTrue(any(item.get("key") == "review-pr" for item in indexables))

    def test_builder_writes_manifest_and_mix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "mix.jsonl"
            manifest = build_prod_extraction_v1(
                output,
                direct_probes=DIRECT_PROBES if DIRECT_PROBES.exists() else None,
                profile={
                    **PROD_EXTRACTION_V1_PROFILE,
                    "plan_copies": 1,
                    "workflow_copies": 1,
                    "technical_copies": 1,
                    "noise_copies": 1,
                    "expanded_copies": 1,
                    "recall_copies": 2,
                    "nano_copies": 0,
                },
            )
            self.assertTrue(output.exists())
            manifest_path = Path(manifest["manifest"])
            self.assertTrue(manifest_path.exists())
            self.assertEqual(manifest["profile"], "prod-extraction-v1")
            self.assertTrue(manifest["validation"]["ok"])
            self.assertGreater(manifest["total_rows"], 0)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
            for row in rows[:20]:
                validate_prod_row(row)

    def test_remember_llm_response_input_shape(self) -> None:
        row = build_fixture_rows()[0]
        operation = row["input"]["operation"]
        conversation = row["input"]["conversation"]
        self.assertEqual(operation, "remember_llm_response")
        self.assertIsInstance(conversation, list)
        self.assertEqual(conversation[0]["role"], "assistant")


if __name__ == "__main__":
    unittest.main()
