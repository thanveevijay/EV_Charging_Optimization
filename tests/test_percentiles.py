import unittest

from evaluation.percentiles import percentile


class PercentileTests(unittest.TestCase):
    def test_median_even_count(self):
        self.assertEqual(percentile([1, 2, 3, 4], 0.5), 2.5)

    def test_ninety_percentile(self):
        self.assertAlmostEqual(percentile([0, 10, 20, 30, 40], 0.9), 36)

    def test_minimum_and_maximum(self):
        self.assertEqual(percentile([3, 1, 2], 0), 1)
        self.assertEqual(percentile([3, 1, 2], 1), 3)

    def test_invalid_quantile(self):
        with self.assertRaises(ValueError):
            percentile([1, 2], 1.1)


if __name__ == "__main__":
    unittest.main()
