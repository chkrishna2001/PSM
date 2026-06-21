import json
import tempfile
import unittest
from pathlib import Path

from prod_memory.build_prod_extraction_v6_storage_only import build_prod_extraction_v6_storage_only
from psm_model.data.rows import infer_row_task

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE = REPO_ROOT / "psm-model" / "prod-memory" / "data" / "prod-extraction-v3.jsonl"


class BuildProdExtractionV6Test(unittest.TestCase):
    def test_storage_only_filter(self) -> None:
        if not SOURCE.exists():
            self.skipTest("prod-extraction-v2.jsonl not present locally")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "prod-extraction-v6.jsonl"
            manifest = build_prod_extraction_v6_storage_only(output, source=SOURCE)
            self.assertEqual(manifest["profile"], "prod-extraction-v6-storage-only")
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertTrue(all(infer_row_task(row) == "storage" for row in rows))
            self.assertGreaterEqual(manifest["input_chars_p50"], 500)


if __name__ == "__main__":
    unittest.main()
