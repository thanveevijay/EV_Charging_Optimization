import unittest

from scheduling.priority import (
    earliest_departure_score,
    energy_need_score,
    combined_priority,
)


class PriorityTests(unittest.TestCase):
    def test_earlier_departure_has_higher_urgency(self):
        self.assertGreater(earliest_departure_score(5), earliest_departure_score(30))

    def test_more_energy_increases_need_score(self):
        self.assertGreater(energy_need_score(20, 7), energy_need_score(5, 7))

    def test_combined_score_is_bounded(self):
        score = combined_priority(10, 12, 7)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_invalid_weight_rejected(self):
        with self.assertRaises(ValueError):
            combined_priority(10, 12, 7, 1.5)


if __name__ == "__main__":
    unittest.main()
