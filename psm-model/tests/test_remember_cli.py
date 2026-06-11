from __future__ import annotations

import json
import unittest

from psm_model.remember_cli import apply_product_boundary, remember_from_repair_payload
from psm_model.schema import validate_storage_decision


class RememberCliBoundaryTests(unittest.TestCase):
    def test_strict_ignore_with_memory_repaired(self) -> None:
        raw = "A:ignore\nT:episodic\nC:No durable memory value.\nEND"
        report = apply_product_boundary(raw, output_format="tagged")
        self.assertTrue(report["valid"])
        self.assertIn(report["repair_status"], {"repaired", "parsed"})
        self.assertEqual(report["parsed"]["action"], "ignore")
        self.assertIsNone(report["parsed"]["memory"])
        self.assertTrue(validate_storage_decision(report["parsed"]).ok)

    def test_word_salad_fails_safe_to_ignore(self) -> None:
        raw = (
            "A: lane the in the minreshary\n"
            "Q:Question,0.02,0.35,0.92\n"
            "G:personamem,preference\n"
            "END"
        )
        report = apply_product_boundary(raw, output_format="tagged")
        self.assertEqual(report["repair_status"], "failed_safe")
        self.assertEqual(report["parsed"]["action"], "ignore")
        self.assertTrue(validate_storage_decision(report["parsed"]).ok)

    def test_repair_payload_accepts_json_invalid_output(self) -> None:
        invalid = json.dumps(
            {
                "action": "ignore",
                "memory": {"content": "x", "type": "episodic"},
                "facts": [],
            }
        )
        report = remember_from_repair_payload(
            {"operation": "repair_remember_json", "invalid_model_output": invalid},
            output_format="tagged",
        )
        self.assertEqual(report["parsed"]["action"], "ignore")
        self.assertIsNone(report["parsed"]["memory"])
        self.assertEqual(report["repair_status"], "repaired")


if __name__ == "__main__":
    unittest.main()
