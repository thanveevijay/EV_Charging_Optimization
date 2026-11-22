import unittest

from evaluation.metrics import (
    waiting_time_summary,
    completion_rate,
    energy_fulfillment_ratio,
    charger_utilization,
)


class EvaluationMetricsTests(unittest.TestCase):
    def test_waiting_time_summary(self):
        result = waiting_time_summary([0, 10, 20, 30])
        self.assertEqual(result["count"], 4.0)
        self.assertEqual(result["mean_minutes"], 15.0)
        self.assertEqual(result["median_minutes"], 15.0)

    def test_waiting_time_rejects_negative_values(self):
        with self.assertRaises(ValueError):
            waiting_time_summary([3, -1])

    def test_completion_rate(self):
        self.assertAlmostEqual(completion_rate(8, 10), 0.8)

    def test_completion_rate_rejects_invalid_counts(self):
        with self.assertRaises(ValueError):
            completion_rate(11, 10)

    def test_energy_fulfillment(self):
        self.assertAlmostEqual(energy_fulfillment_ratio(18, 20), 0.9)

    def test_energy_fulfillment_requires_positive_request(self):
        with self.assertRaises(ValueError):
            energy_fulfillment_ratio(0, 0)

    def test_charger_utilization(self):
        self.assertAlmostEqual(charger_utilization(30, 60), 0.5)

    def test_charger_utilization_rejects_over_capacity(self):
        with self.assertRaises(ValueError):
            charger_utilization(61, 60)


if __name__ == "__main__":
    unittest.main()
