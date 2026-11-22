import unittest
from scheduling.action_mask import available_action_mask, validate_action


class ActionMaskTests(unittest.TestCase):
    def test_mask_conversion(self):
        self.assertEqual(available_action_mask([True, False, True]), [1, 0, 1])

    def test_action_validation(self):
        mask = [1, 0, 1]
        self.assertTrue(validate_action(2, mask))
        self.assertFalse(validate_action(1, mask))
        self.assertFalse(validate_action(3, mask))


if __name__ == "__main__":
    unittest.main()
