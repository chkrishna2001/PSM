import unittest

from psm_model.gates import DIRECT_PROBE_THRESHOLDS, evaluate_thresholds, gate_report


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


if __name__ == "__main__":
    unittest.main()
