import unittest
from evaluation.demand import demand_summary


class DemandSummaryTests(unittest.TestCase):
    def test_summary(self):
        result = demand_summary([1, 3], [5.0, 7.0])
        self.assertEqual(result["total_arrivals"], 4.0)
        self.assertEqual(result["total_energy_kwh"], 12.0)

    def test_rejects_mismatched_series(self):
        with self.assertRaises(ValueError):
            demand_summary([1], [2, 3])


if __name__ == "__main__":
    unittest.main()
