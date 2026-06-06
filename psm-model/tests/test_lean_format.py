import json
import unittest
from pathlib import Path

from psm_model.compare_formats import compare_file
from psm_model.lean_format import (
    compact_json_array,
    encode_at_tag_decision,
    encode_tagged_decision,
    parse_at_tag_decision,
    parse_tagged_decision,
)


class LeanFormatTests(unittest.TestCase):
    def test_tagged_format_round_trips_direct_probes(self):
        path = Path(__file__).resolve().parents[1] / "data" / "probes" / "direct_probes.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

        for row in rows:
            with self.subTest(row=row["id"]):
                encoded = encode_tagged_decision(row["expected"])
                parsed, issues = parse_tagged_decision(encoded)
                self.assertEqual(issues, ())
                self.assertIsNotNone(parsed)
                self.assertEqual(parsed["action"], row["expected"]["action"])

    def test_at_tag_format_round_trips_direct_probes(self):
        path = Path(__file__).resolve().parents[1] / "data" / "probes" / "direct_probes.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

        for row in rows:
            with self.subTest(row=row["id"]):
                encoded = encode_at_tag_decision(row["expected"])
                parsed, issues = parse_at_tag_decision(encoded)
                self.assertEqual(issues, ())
                self.assertIsNotNone(parsed)
                self.assertEqual(parsed["action"], row["expected"]["action"])
                self.assertIn("@end", encoded)

    def test_compact_array_is_valid_json(self):
        decision = {
            "action": "ignore",
            "memory": None,
            "facts": [],
            "reasoning": "No durable memory.",
        }

        encoded = compact_json_array(decision)

        self.assertEqual(json.loads(encoded), ["ignore", None, [], "No durable memory."])

    def test_pipe_tagged_format_escapes_free_text_delimiters(self):
        decision = {
            "action": "promote_semantic",
            "memory": {
                "content": "The user likes A|B formats, but wants backslash \\ handling.\nKeep exact evidence.",
                "type": "semantic",
                "strength": 0.8,
                "decay_rate": 0.02,
                "emotional_weight": 0.1,
                "confidence": 0.9,
                "tags": ["format_test", "pipe"],
            },
            "facts": [
                {
                    "subject": "user",
                    "predicate": "likes",
                    "value": "A|B formats, with backslash \\ handling",
                    "confidence": 0.9,
                    "inference_kind": "explicit",
                    "evidence_text": "I like A|B formats, but backslash \\ must work.\nThis is evidence.",
                }
            ],
            "reasoning": "Delimiter-heavy content should round-trip.",
        }

        encoded = encode_tagged_decision(decision)
        parsed, issues = parse_tagged_decision(encoded)

        self.assertEqual(issues, ())
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["memory"]["content"], decision["memory"]["content"])
        self.assertEqual(parsed["facts"][0]["value"], decision["facts"][0]["value"])
        self.assertEqual(parsed["facts"][0]["evidence_text"], decision["facts"][0]["evidence_text"])

    def test_tagged_parser_merges_extra_fact_pipes_into_evidence(self):
        raw = "\n".join(
            [
                "A:promote_semantic",
                "T:semantic",
                "C:The user prefers concise answers.",
                "Q:0.85,0.02,0.35,0.9",
                "F:User|prefers|concise answers|0.96|explicit|I prefer concise|technical answers",
                "R:Explicit durable preference.",
                "END",
            ]
        )

        parsed, issues = parse_tagged_decision(raw)

        self.assertEqual(issues, ())
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["facts"][0]["evidence_text"], "I prefer concise|technical answers")

    def test_compare_file_reports_savings(self):
        path = Path(__file__).resolve().parents[1] / "data" / "probes" / "direct_probes.jsonl"

        report = compare_file(path)

        self.assertEqual(report["rows"], 5)
        self.assertGreater(report["totals"]["json"], report["totals"]["tagged"])
        self.assertGreater(report["totals"]["json"], report["totals"]["at_tag"])
        self.assertGreater(report["tagged_total_savings_vs_json"], 0)


if __name__ == "__main__":
    unittest.main()
