import unittest
from evaluation.energy import kw_minutes_to_kwh, fulfillment_percentage


class EnergyMetricTests(unittest.TestCase):
    def test_power_time_conversion(self):
        self.assertAlmostEqual(kw_minutes_to_kwh(7.2, 5), 0.6)

    def test_fulfillment_percentage(self):
        self.assertEqual(fulfillment_percentage(8, 10), 80.0)

    def test_invalid_request_rejected(self):
        with self.assertRaises(ValueError):
            fulfillment_percentage(1, 0)


if __name__ == "__main__":
    unittest.main()
