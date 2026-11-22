import unittest
from evaluation.fairness import grouped_mean_wait


class FairnessTests(unittest.TestCase):
    def test_grouped_wait_means(self):
        self.assertEqual(
            grouped_mean_wait(["day", "night", "day"], [10, 20, 30]),
            {"day": 20.0, "night": 20.0},
        )

    def test_mismatched_lengths_rejected(self):
        with self.assertRaises(ValueError):
            grouped_mean_wait(["a"], [1, 2])


if __name__ == "__main__":
    unittest.main()
