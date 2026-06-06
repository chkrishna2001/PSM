import unittest

from psm_model.gate_checkpoint import (
    PHASE1_ACTION_THRESHOLDS,
    PHASE1_MIN_DISTINCT_ACTIONS,
    compact_action_report,
    compact_head_report,
    compact_manual_report,
    evaluate_gate_metrics,
    thresholds_for_mode,
)


class GateCheckpointTests(unittest.TestCase):
    def test_evaluate_gate_metrics_fails_missing_and_low_values(self):
        checks = evaluate_gate_metrics(
            {"foundation_macro_action_prefix_accuracy": 0.91, "concept_macro_action_prefix_accuracy": 0.8},
            {
                "foundation_macro_action_prefix_accuracy": 0.9,
                "concept_macro_action_prefix_accuracy": 0.85,
                "manual_model_action_accuracy": 0.7,
            },
        )

        failures = [check.metric for check in checks if not check.passed]

        self.assertEqual(failures, ["concept_macro_action_prefix_accuracy", "manual_model_action_accuracy"])

    def test_compact_action_report_keeps_failures_without_full_scores(self):
        report = {
            "data": "probe.jsonl",
            "examples": 2,
            "macro_action_prefix_accuracy": 0.5,
            "per_action_accuracy": {"ignore": 1.0, "promote_semantic": 0.0},
            "predicted_action_counts": {"ignore": 2},
            "reports": [
                {"id": "ok", "expected_action": "ignore", "predicted_action": "ignore", "gold_rank": 1},
                {"id": "bad", "expected_action": "promote_semantic", "predicted_action": "ignore", "gold_rank": 2},
            ],
        }

        compact = compact_action_report(report)

        self.assertEqual(compact["failures"], [{"id": "bad", "expected_action": "promote_semantic", "predicted_action": "ignore", "gold_rank": 2}])

    def test_compact_manual_report_separates_safe_failures_from_model_misses(self):
        report = {
            "examples": 1,
            "expected_action_accuracy": 1.0,
            "model_action_accuracy": 0.0,
            "valid_rate": 1.0,
            "reports": [
                {
                    "case": "preference",
                    "expected_action": "promote_semantic",
                    "model_action": "store_episodic",
                    "calibrated_action": "promote_semantic",
                    "parsed_action": "promote_semantic",
                    "valid": True,
                }
            ],
        }

        compact = compact_manual_report(report)

        self.assertEqual(compact["failures"], [])
        self.assertEqual(compact["model_action_misses"][0]["model_action"], "store_episodic")

    def test_compact_head_report_keeps_action_head_failures(self):
        report = {
            "data": "manual.jsonl",
            "examples": 2,
            "action_head_accuracy": 0.5,
            "macro_action_head_accuracy": 0.5,
            "per_action_accuracy": {"ignore": 1.0, "promote_semantic": 0.0},
            "predicted_action_counts": {"ignore": 2},
            "reports": [
                {"id": "ok", "expected_action": "ignore", "predicted_action": "ignore"},
                {"id": "bad", "expected_action": "promote_semantic", "predicted_action": "ignore"},
            ],
        }

        compact = compact_head_report(report)

        self.assertEqual(compact["action_head_accuracy"], 0.5)
        self.assertEqual(compact["failures"], [{"id": "bad", "expected_action": "promote_semantic", "predicted_action": "ignore"}])

    def test_product_safe_thresholds_exclude_model_action_selector(self):
        thresholds = thresholds_for_mode("product-safe")

        self.assertEqual(
            thresholds,
            {
                "manual_safe_expected_action_accuracy": 1.0,
                "manual_safe_valid_rate": 1.0,
            },
        )

    def test_phase1_action_thresholds(self):
        self.assertEqual(thresholds_for_mode("phase1-action"), PHASE1_ACTION_THRESHOLDS)
        self.assertEqual(PHASE1_MIN_DISTINCT_ACTIONS, 4)

    def test_phase1_distinct_action_failure(self):
        metrics = {
            "expanded_macro_action_prefix_accuracy": 0.9,
            "manual_macro_action_prefix_accuracy": 0.9,
        }
        checks = evaluate_gate_metrics(metrics, PHASE1_ACTION_THRESHOLDS)
        distinct_predicted = 2
        passed = all(check.passed for check in checks) and distinct_predicted >= PHASE1_MIN_DISTINCT_ACTIONS
        self.assertFalse(passed)


if __name__ == "__main__":
    unittest.main()
