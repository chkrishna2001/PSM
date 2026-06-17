import unittest

from prod_memory.grounding import (
    apply_storage_guards,
    grounding_overlap_score,
    has_curriculum_bleed,
    is_grounded_in_source,
)


class GroundingMetricsTest(unittest.TestCase):
    def test_bleed_detects_gate_curriculum(self) -> None:
        self.assertTrue(has_curriculum_bleed("Today run constoursated fact parser"))
        self.assertTrue(has_curriculum_bleed("Resume from checkpoint step-062000"))

    def test_grounded_when_overlap_exists(self) -> None:
        target = "Review pull request changes before approve."
        stored = "Procedure: review changed files in the pull request."
        self.assertTrue(is_grounded_in_source(target, stored))

    def test_guard_rejects_ungrounded_store(self) -> None:
        decision = {
            "action": "store_episodic",
            "memory": {"content": "User prefers SQLite for local apps."},
            "facts": [],
        }
        guarded = apply_storage_guards("RunPod verify tmux GPU training policy.", decision)
        self.assertTrue(guarded["rejected"])
        self.assertEqual(guarded["route"], "grounding_reject")

    def test_guard_allows_grounded_store(self) -> None:
        decision = {
            "action": "store_episodic",
            "memory": {"content": "Review pull request changes and approve when scope is correct."},
            "facts": [],
        }
        guarded = apply_storage_guards("Review pull request changes before approve.", decision)
        self.assertFalse(guarded["rejected"])

    def test_overlap_score_threshold(self) -> None:
        score = grounding_overlap_score("chunking markdown remember context", "Uses chunking on markdown sections.")
        self.assertGreaterEqual(score["overlap"], score["required"])


if __name__ == "__main__":
    unittest.main()
