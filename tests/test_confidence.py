import unittest
from evaluation.confidence import mean_confidence_interval


class ConfidenceIntervalTests(unittest.TestCase):
    def test_interval_contains_mean(self):
        low, high = mean_confidence_interval([1, 2, 3, 4, 5])
        self.assertLessEqual(low, 3)
        self.assertGreaterEqual(high, 3)

    def test_requires_multiple_runs(self):
        with self.assertRaises(ValueError):
            mean_confidence_interval([1])


if __name__ == "__main__":
    unittest.main()
