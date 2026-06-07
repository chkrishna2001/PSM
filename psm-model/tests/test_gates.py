import unittest

from psm_model.analyze_eval_report import analyze_eval_report, classify_row
from psm_model.gates import (
    DIRECT_PROBE_THRESHOLDS,
    EXPANDED_PROBE_THRESHOLDS,
    evaluate_thresholds,
    gate_report,
    thresholds_for_gate_mode,
)


class GateTests(unittest.TestCase):
    def test_direct_probe_gate_requires_exact_generated_outputs(self):
        report = {metric: 1.0 for metric in DIRECT_PROBE_THRESHOLDS}
        report["facts_exact_rate"] = 0.8

        failures = evaluate_thresholds(report)

        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].metric, "facts_exact_rate")

    def test_gate_report_passes_when_all_thresholds_met(self):
        report = {metric: 1.0 for metric in DIRECT_PROBE_THRESHOLDS}

        result = gate_report(report)

        self.assertTrue(result["passed"])
        self.assertEqual(result["failures"], [])

    def test_expanded_probe_gate_allows_relaxed_metrics(self):
        report = {metric: required for metric, required in EXPANDED_PROBE_THRESHOLDS.items()}

        result = gate_report(report, EXPANDED_PROBE_THRESHOLDS)

        self.assertTrue(result["passed"])
        self.assertEqual(result["failures"], [])

    def test_expanded_probe_gate_fails_below_action_bar(self):
        report = {metric: required for metric, required in EXPANDED_PROBE_THRESHOLDS.items()}
        report["action_accuracy"] = 0.84

        failures = evaluate_thresholds(report, EXPANDED_PROBE_THRESHOLDS)

        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].metric, "action_accuracy")

    def test_thresholds_for_gate_mode(self):
        self.assertEqual(thresholds_for_gate_mode("direct"), DIRECT_PROBE_THRESHOLDS)
        self.assertEqual(thresholds_for_gate_mode("expanded"), EXPANDED_PROBE_THRESHOLDS)

    def test_classify_row_buckets_failures(self):
        self.assertEqual(classify_row({"skipped": True}), "context_overflow")
        self.assertEqual(
            classify_row(
                {
                    "parse_valid": True,
                    "schema_valid": True,
                    "expected_action": "promote_semantic",
                    "predicted_action": "ignore",
                }
            ),
            "wrong_action",
        )

    def test_analyze_eval_report_summarizes_rows(self):
        eval_report = {
            "checkpoint": "ckpt.pt",
            "parse_valid_rate": 0.5,
            "schema_valid_rate": 0.5,
            "action_accuracy": 0.5,
            "memory_type_accuracy": 0.5,
            "memory_content_exact_rate": 0.5,
            "fact_count_accuracy": 0.5,
            "facts_exact_rate": 0.5,
            "reports": [
                {
                    "id": "a",
                    "parse_valid": False,
                    "schema_valid": False,
                    "expected_action": "promote_semantic",
                },
                {
                    "id": "b",
                    "parse_valid": True,
                    "schema_valid": True,
                    "expected_action": "promote_semantic",
                    "predicted_action": "promote_semantic",
                    "expected_memory_type": "semantic",
                    "predicted_memory_type": "semantic",
                    "memory_content_exact": True,
                    "expected_fact_count": 0,
                    "predicted_fact_count": 0,
                    "facts_exact": True,
                },
            ],
        }

        analysis = analyze_eval_report(eval_report, gate_mode="expanded")

        self.assertEqual(analysis["bucket_counts"]["parse_fail"], 1)
        self.assertEqual(analysis["bucket_counts"]["pass"], 1)
        self.assertFalse(analysis["gate"]["passed"])


if __name__ == "__main__":
    unittest.main()
