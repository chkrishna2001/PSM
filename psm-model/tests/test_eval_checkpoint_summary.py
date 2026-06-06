from __future__ import annotations

import unittest
from unittest.mock import patch

from psm_model.eval_checkpoint_summary import summarize_checkpoint


class EvalCheckpointSummaryTests(unittest.TestCase):
    def test_summary_keeps_compact_gate_and_raw_metrics(self) -> None:
        gate = {
            "passed": False,
            "failures": [{"metric": "x", "actual": 0.0, "required": 1.0}],
            "metrics": {
                "manual_action_head_accuracy": 0.5,
                "manual_macro_action_head_accuracy": 0.4,
                "manual_safe_expected_action_accuracy": 1.0,
                "manual_safe_valid_rate": 1.0,
                "manual_model_action_accuracy": 0.8,
            },
            "details": {
                "foundation": {"failures": [1]},
                "concept": {"failures": [1, 2]},
                "fast_mixed": {"failures": []},
                "manual_action_head": {"predicted_action_counts": {"ignore": 2}, "failures": [1, 2, 3]},
            },
        }
        raw = {"expected_action_accuracy": 0.25, "valid_rate": 0.75}

        with patch("psm_model.eval_checkpoint_summary.evaluate_checkpoint_gate", return_value=gate), patch(
            "psm_model.eval_checkpoint_summary._load_probe_file", return_value=[]
        ), patch("psm_model.eval_checkpoint_summary.probe_checkpoint", return_value=raw):
            summary = summarize_checkpoint("checkpoint.pt")  # type: ignore[arg-type]

        self.assertFalse(summary["passed"])
        self.assertEqual(summary["prefix_failures"]["concept"], 2)
        self.assertEqual(summary["manual_action_head"]["failure_count"], 3)
        self.assertEqual(summary["manual_raw"]["valid_rate"], 0.75)


if __name__ == "__main__":
    unittest.main()
