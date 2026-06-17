import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from prod_memory.hf_assets import (
    CURRICULUM_REL,
    DEFAULT_DATASET_REPO,
    RESUME_STEP,
    SMOKE_TARGET_STEPS,
)
from prod_memory.upload_hf import upload_prod_dataset


class UploadProdHfTest(unittest.TestCase):
    def test_upload_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            curriculum = root / "mix.jsonl"
            manifest = root / "mix.manifest.json"
            curriculum.write_text('{"id":"row"}\n', encoding="utf-8")
            manifest.write_text("{}", encoding="utf-8")
            report = upload_prod_dataset(
                repo_id=DEFAULT_DATASET_REPO,
                curriculum=curriculum,
                manifest=manifest,
                dry_run=True,
            )
            self.assertTrue(report["dry_run"])
            remotes = {item["remote"] for item in report["files"]}
            self.assertIn(CURRICULUM_REL, remotes)

    @patch("huggingface_hub.HfApi")
    def test_upload_calls_hf_api(self, hf_api_cls: MagicMock) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            curriculum = root / "mix.jsonl"
            manifest = root / "mix.manifest.json"
            curriculum.write_text('{"id":"row"}\n', encoding="utf-8")
            manifest.write_text("{}", encoding="utf-8")
            api = MagicMock()
            hf_api_cls.return_value = api
            report = upload_prod_dataset(
                repo_id=DEFAULT_DATASET_REPO,
                curriculum=curriculum,
                manifest=manifest,
            )
            self.assertEqual(report["uploaded"], [CURRICULUM_REL, "prod-memory/prod-extraction-v1.manifest.json"])
            self.assertEqual(api.upload_file.call_count, 2)


class HfAssetsTest(unittest.TestCase):
    def test_smoke_target_steps(self) -> None:
        self.assertEqual(SMOKE_TARGET_STEPS, RESUME_STEP + 2000)


if __name__ == "__main__":
    unittest.main()
