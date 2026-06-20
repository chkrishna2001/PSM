import json
import tempfile
import unittest
from pathlib import Path

from prod_memory.build_prod_extraction_v2 import PROD_EXTRACTION_V2_PROFILE, build_prod_extraction_v2
from prod_memory.ingest_training_data import ingest_training_directory
from prod_memory.label_from_assistant import build_expected_from_assistant, extract_facts, extract_memory_content
from prod_memory.row_validation import validate_prod_row
from prod_memory.segment_text import segment_llm_response

REPO_ROOT = Path(__file__).resolve().parents[3]
DIRECT_PROBES = REPO_ROOT / "psm-model" / "data" / "probes" / "direct_probes.jsonl"
TRAINING_DATA = Path.home() / "Downloads" / "training-data"

SAMPLE_ASSISTANT = """## Auth rollout

Ship OAuth refresh tokens with 15-minute access TTL.

1. Add refresh token table in SQLite migrations.
2. Update login handler to rotate tokens on each refresh.
3. Document migration steps in the deployment runbook.

Always use explicit return types on exported TypeScript helpers.
"""


class BuildProdExtractionV2Test(unittest.TestCase):
    def test_segment_and_label_long_assistant_text(self) -> None:
        segments = segment_llm_response(SAMPLE_ASSISTANT)
        self.assertGreaterEqual(len(segments), 1)
        content = extract_memory_content(SAMPLE_ASSISTANT)
        self.assertTrue("OAuth" in content or "Auth rollout" in content)
        facts = extract_facts(SAMPLE_ASSISTANT)
        self.assertGreaterEqual(len(facts), 1)
        expected = build_expected_from_assistant(SAMPLE_ASSISTANT)
        self.assertIsNotNone(expected)
        assert expected is not None
        self.assertNotEqual(expected["action"], "ignore")
        self.assertTrue(expected["facts"])

    def test_v2_profile_uses_single_copies_for_primary_buckets(self) -> None:
        self.assertEqual(PROD_EXTRACTION_V2_PROFILE["plan_copies"], 1)
        self.assertEqual(PROD_EXTRACTION_V2_PROFILE["workflow_copies"], 1)
        self.assertEqual(PROD_EXTRACTION_V2_PROFILE["recall_copies"], 50)

    def test_builder_writes_manifest_with_session_rows_when_data_present(self) -> None:
        if not TRAINING_DATA.exists():
            self.skipTest("training-data folder not present")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "mix.jsonl"
            manifest = build_prod_extraction_v2(
                output,
                training_data_root=TRAINING_DATA,
                direct_probes=DIRECT_PROBES if DIRECT_PROBES.exists() else None,
                context_length=4096,
                profile={
                    **PROD_EXTRACTION_V2_PROFILE,
                    "recall_copies": 2,
                    "expanded_copies": 1,
                },
            )
            self.assertTrue(output.exists())
            self.assertEqual(manifest["profile"], "prod-extraction-v2")
            self.assertGreater(manifest["added"]["session"], 0)
            self.assertGreater(manifest["storage_stats"]["rows_with_facts"], 0)
            self.assertGreater(manifest["storage_stats"]["input_chars_p50"], 400)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
            session_rows = [row for row in rows if str(row.get("id", "")).startswith("session:")]
            self.assertGreater(len(session_rows), 0)
            for row in session_rows[:10]:
                validate_prod_row(row)

    def test_ingest_training_directory_deduplicates(self) -> None:
        if not TRAINING_DATA.exists():
            self.skipTest("training-data folder not present")
        rows, report = ingest_training_directory(TRAINING_DATA)
        self.assertGreater(report["accepted"], 0)
        self.assertGreaterEqual(report["rows"], len(rows))


if __name__ == "__main__":
    unittest.main()
