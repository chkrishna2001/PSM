import json
from pathlib import Path
import unittest

from psm_model.schema import parse_and_validate_storage_decision, validate_storage_decision


VALID_DECISION = {
    "action": "promote_semantic",
    "memory": {
        "content": "The user prefers SQLite for local prototypes.",
        "type": "semantic",
        "strength": 0.8,
        "decay_rate": 0.02,
        "emotional_weight": 0.1,
        "confidence": 0.9,
        "tags": ["preference", "sqlite"],
    },
    "facts": [
        {
            "subject": "user",
            "predicate": "prefers",
            "value": "SQLite for local prototypes",
            "confidence": 0.9,
            "inference_kind": "explicit",
            "evidence_text": "I prefer SQLite for local prototypes.",
        }
    ],
    "reasoning": "The user stated a durable preference.",
}


class SchemaValidationTests(unittest.TestCase):
    def test_valid_storage_decision_passes(self):
        result = validate_storage_decision(VALID_DECISION)

        self.assertTrue(result.ok, result.issues)
        self.assertIsNotNone(result.decision)
        self.assertEqual(result.decision.action, "promote_semantic")

    def test_parse_rejects_malformed_json(self):
        result = parse_and_validate_storage_decision("not json")

        self.assertFalse(result.ok)
        self.assertEqual(result.issues[0].path, "$")
        self.assertIn("invalid JSON", result.issues[0].message)

    def test_unknown_action_fails(self):
        decision = {**VALID_DECISION, "action": "store"}
        result = validate_storage_decision(decision)

        self.assertFalse(result.ok)
        self.assertIssue(result, "$.action")

    def test_fact_without_evidence_fails(self):
        decision = json.loads(json.dumps(VALID_DECISION))
        del decision["facts"][0]["evidence_text"]

        result = validate_storage_decision(decision)

        self.assertFalse(result.ok)
        self.assertIssue(result, "$.facts[0].evidence_text")

    def test_ignore_requires_null_memory(self):
        decision = {**VALID_DECISION, "action": "ignore"}

        result = validate_storage_decision(decision)

        self.assertFalse(result.ok)
        self.assertIssue(result, "$.memory")

    def test_non_ignore_requires_memory(self):
        decision = {**VALID_DECISION, "memory": None}

        result = validate_storage_decision(decision)

        self.assertFalse(result.ok)
        self.assertIssue(result, "$.memory")

    def test_fact_predicate_must_be_snake_case(self):
        decision = json.loads(json.dumps(VALID_DECISION))
        decision["facts"][0]["predicate"] = "Prefers Database"

        result = validate_storage_decision(decision)

        self.assertFalse(result.ok)
        self.assertIssue(result, "$.facts[0].predicate")

    def test_direct_probe_expected_outputs_pass(self):
        probe_path = Path(__file__).resolve().parents[1] / "data" / "probes" / "direct_probes.jsonl"

        with probe_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                with self.subTest(row=row["id"]):
                    result = validate_storage_decision(row["expected"])
                    self.assertTrue(result.ok, result.issues)

    def test_non_string_memory_tags_fail(self):
        decision = json.loads(json.dumps(VALID_DECISION))
        decision["memory"]["tags"] = ["preference", 42]

        result = validate_storage_decision(decision)

        self.assertFalse(result.ok)
        self.assertIssue(result, "$.memory.tags")

    def test_inferred_facts_fail(self):
        decision = json.loads(json.dumps(VALID_DECISION))
        decision["facts"][0]["inference_kind"] = "inferred"

        result = validate_storage_decision(decision)

        self.assertFalse(result.ok)
        self.assertIssue(result, "$.facts[0].inference_kind")

    def assertIssue(self, result, path):
        self.assertIn(path, [issue.path for issue in result.issues])


if __name__ == "__main__":
    unittest.main()
