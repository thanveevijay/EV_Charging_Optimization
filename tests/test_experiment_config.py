import unittest
from scripts.check_experiment_config import validate_config


class ExperimentConfigTests(unittest.TestCase):
    def test_valid_config(self):
        data = {
            "experiment": {"name": "demo"},
            "simulation": {"charger_power_kw": 7.2, "time_step_minutes": 5},
            "policy": {"baseline": "FCFS"},
        }
        self.assertEqual(validate_config(data), [])

    def test_missing_sections(self):
        self.assertTrue(validate_config({}))

    def test_non_positive_power_rejected(self):
        data = {
            "experiment": {},
            "simulation": {"charger_power_kw": 0, "time_step_minutes": 5},
            "policy": {},
        }
        self.assertTrue(validate_config(data))


if __name__ == "__main__":
    unittest.main()
