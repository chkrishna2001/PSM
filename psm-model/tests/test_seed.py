import unittest
from collections import Counter

from psm_model.data.seed import generate_seed_rows, split_rows
from psm_model.data.rows import validate_training_row


class SeedDatasetTests(unittest.TestCase):
    def test_seed_rows_are_valid_and_balanced(self):
        rows = generate_seed_rows()
        actions = Counter(row["expected"]["action"] for row in rows)

        self.assertEqual(len(rows), 60)
        self.assertEqual(set(actions), {"ignore", "store_episodic", "promote_semantic", "update_existing", "flag_and_store"})
        self.assertGreaterEqual(actions["ignore"], 10)
        for row in rows:
            with self.subTest(row=row["id"]):
                _, issues = validate_training_row(row)
                self.assertEqual(issues, ())

    def test_seed_split_is_deterministic(self):
        rows = generate_seed_rows()
        train, validation = split_rows(rows, validation_every=5)

        self.assertEqual(len(train), 48)
        self.assertEqual(len(validation), 12)
        self.assertEqual(validation[0]["id"], rows[4]["id"])


if __name__ == "__main__":
    unittest.main()

