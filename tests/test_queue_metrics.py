import unittest

from scheduling.queue_metrics import average_queue_length, maximum_queue_length


class QueueMetricTests(unittest.TestCase):
    def test_average_queue(self):
        self.assertEqual(average_queue_length([0, 2, 4]), 2)

    def test_maximum_queue(self):
        self.assertEqual(maximum_queue_length([1, 3, 2]), 3)

    def test_empty_observations_rejected(self):
        with self.assertRaises(ValueError):
            average_queue_length([])

    def test_negative_queue_rejected(self):
        with self.assertRaises(ValueError):
            maximum_queue_length([1, -2])


if __name__ == "__main__":
    unittest.main()
