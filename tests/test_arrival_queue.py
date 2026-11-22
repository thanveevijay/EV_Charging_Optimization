import unittest
from scheduling.arrival_queue import ArrivalQueue


class ArrivalQueueTests(unittest.TestCase):
    def test_fifo_order(self):
        queue = ArrivalQueue()
        queue.add("ev-1")
        queue.add("ev-2")
        self.assertEqual(queue.pop_next(), "ev-1")
        self.assertEqual(queue.pop_next(), "ev-2")

    def test_empty_pop_raises(self):
        with self.assertRaises(IndexError):
            ArrivalQueue().pop_next()


if __name__ == "__main__":
    unittest.main()
