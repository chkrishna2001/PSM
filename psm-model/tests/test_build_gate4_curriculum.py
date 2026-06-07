import json
import tempfile
import unittest
from pathlib import Path

from psm_model.build_gate4_curriculum import build_gate4_curriculum


class BuildGate4CurriculumTests(unittest.TestCase):
    def test_builds_anchor_and_ignore_oversample(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.jsonl"
            direct = root / "direct.jsonl"
            expanded = root / "expanded.jsonl"
            output = root / "out.jsonl"

            base.write_text(
                json.dumps(
                    {
                        "id": "base-1",
                        "input": {"conversation": "User: base row"},
                        "expected": {"action": "promote_semantic", "memory": None, "facts": []},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            direct.write_text(
                json.dumps(
                    {
                        "id": "direct-1",
                        "input": {"conversation": "User: direct"},
                        "expected": {"action": "ignore", "memory": None, "facts": []},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            expanded.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "id": f"expanded-{index}",
                            "input": {"conversation": f"User: expanded {index}"},
                            "expected": {
                                "action": "ignore" if index == 0 else "promote_semantic",
                                "memory": None,
                                "facts": [],
                            },
                        }
                    )
                    for index in range(2)
                )
                + "\n",
                encoding="utf-8",
            )

            summary = build_gate4_curriculum(
                base,
                output,
                direct_probes=direct,
                expanded_probes=expanded,
                direct_copies=2,
                expanded_copies=2,
                ignore_extra_copies=3,
            )

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(summary["base_rows"], 1)
            self.assertEqual(summary["direct_anchor_rows"], 2)
            self.assertEqual(summary["expanded_anchor_rows"], 4)
            self.assertEqual(summary["ignore_extra_rows"], 3)
            self.assertEqual(len(rows), 10)
            self.assertTrue(any(row["id"].startswith("expanded-ignore:") for row in rows))


if __name__ == "__main__":
    unittest.main()
