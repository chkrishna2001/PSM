import json
import tempfile
import unittest
from pathlib import Path

from prod_memory.build_prod_extraction_v7_storage_only import (
    build_prod_extraction_v7_storage_only,
    normalize_expected_action,
)
from psm_model.data.rows import infer_row_task

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE = REPO_ROOT / "psm-model" / "prod-memory" / "data" / "prod-extraction-v3.jsonl"


class BuildProdExtractionV7Test(unittest.TestCase):
    def test_flag_and_store_maps_to_promote_semantic(self) -> None:
        expected = normalize_expected_action(
            {"action": "flag_and_store", "memory": {"type": "semantic", "content": "x"}}
        )
        self.assertEqual(expected["action"], "promote_semantic")

    def test_storage_only_and_eval_actions(self) -> None:
        if not SOURCE.exists():
            self.skipTest("prod-extraction-v3.jsonl not present locally")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "prod-extraction-v7.jsonl"
            manifest = build_prod_extraction_v7_storage_only(output, source=SOURCE)
            self.assertEqual(manifest["profile"], "prod-extraction-v7-storage-only")
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertTrue(all(infer_row_task(row) == "storage" for row in rows))
            actions = {row["expected"]["action"] for row in rows}
            self.assertTrue(actions <= {"ignore", "store_episodic", "promote_semantic"})
            self.assertGreaterEqual(manifest["input_chars_p50"], 500)


if __name__ == "__main__":
    unittest.main()
