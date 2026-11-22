import unittest

from evaluation.statistics import mean, sample_std, mean_and_std


class StatisticsTests(unittest.TestCase):
    def test_mean(self):
        self.assertEqual(mean([1, 2, 3]), 2)

    def test_sample_std(self):
        self.assertAlmostEqual(sample_std([1, 2, 3]), 1.0)

    def test_summary(self):
        result = mean_and_std([2, 4, 6])
        self.assertEqual(result["mean"], 4)
        self.assertEqual(result["runs"], 3.0)

    def test_empty_mean_rejected(self):
        with self.assertRaises(ValueError):
            mean([])

    def test_single_run_std_rejected(self):
        with self.assertRaises(ValueError):
            sample_std([5])


if __name__ == "__main__":
    unittest.main()
