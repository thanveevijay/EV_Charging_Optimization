import unittest

from scheduling.constraints import can_assign


class AssignmentConstraintTests(unittest.TestCase):
    def test_valid_assignment(self):
        self.assertTrue(can_assign(True, True, 10, 7, 20))

    def test_unavailable_charger(self):
        self.assertFalse(can_assign(False, True, 0, 7, 20))

    def test_incompatible_vehicle(self):
        self.assertFalse(can_assign(True, False, 0, 7, 20))

    def test_station_limit(self):
        self.assertFalse(can_assign(True, True, 15, 7, 20))

    def test_negative_power_rejected(self):
        with self.assertRaises(ValueError):
            can_assign(True, True, -1, 7, 20)


if __name__ == "__main__":
    unittest.main()
