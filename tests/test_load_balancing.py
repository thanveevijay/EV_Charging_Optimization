import unittest
from scheduling.load_balancing import remaining_station_capacity, clamp_requested_power


class LoadBudgetTests(unittest.TestCase):
    def test_remaining_capacity(self):
        self.assertEqual(remaining_station_capacity(30, 12), 18)

    def test_power_is_clamped_by_lowest_limit(self):
        self.assertEqual(clamp_requested_power(10, 7, 6, 4), 4)


if __name__ == "__main__":
    unittest.main()
