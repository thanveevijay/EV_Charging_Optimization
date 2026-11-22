import unittest
from evaluation.waiting import waiting_rate, zero_wait_share


class WaitingMetricTests(unittest.TestCase):
    def test_waiting_rate(self):
        self.assertEqual(waiting_rate([0, 5, 20], 10), 1 / 3)

    def test_zero_wait_share(self):
        self.assertEqual(zero_wait_share([0, 0, 10]), 2 / 3)


if __name__ == "__main__":
    unittest.main()
