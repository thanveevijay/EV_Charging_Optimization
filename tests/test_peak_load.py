import unittest

from evaluation.peak_load import peak_load_kw, average_load_kw, load_factor


class PeakLoadTests(unittest.TestCase):
    def test_peak(self):
        self.assertEqual(peak_load_kw([1, 4, 2]), 4)

    def test_average(self):
        self.assertAlmostEqual(average_load_kw([2, 4, 6]), 4)

    def test_load_factor(self):
        self.assertAlmostEqual(load_factor([2, 4, 6]), 2 / 3)

    def test_zero_load_factor(self):
        self.assertEqual(load_factor([0, 0]), 0)

    def test_negative_load_rejected(self):
        with self.assertRaises(ValueError):
            peak_load_kw([1, -1])


if __name__ == "__main__":
    unittest.main()
