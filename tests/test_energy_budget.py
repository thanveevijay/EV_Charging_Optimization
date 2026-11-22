import unittest

from scheduling.energy_budget import maximum_step_energy_kwh, energy_shortfall_kwh


class EnergyBudgetTests(unittest.TestCase):
    def test_charger_step_energy(self):
        self.assertAlmostEqual(maximum_step_energy_kwh(7.2, 5), 0.6)

    def test_vehicle_acceptance_limits_energy(self):
        self.assertAlmostEqual(maximum_step_energy_kwh(11, 10, 3.6), 0.6)

    def test_invalid_power_rejected(self):
        with self.assertRaises(ValueError):
            maximum_step_energy_kwh(0, 5)

    def test_shortfall(self):
        self.assertEqual(energy_shortfall_kwh(20, 15), 5)

    def test_no_negative_shortfall(self):
        self.assertEqual(energy_shortfall_kwh(10, 12), 0)


if __name__ == "__main__":
    unittest.main()
