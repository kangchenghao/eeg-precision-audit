import unittest

from budget_ceiling import best_static_mixed


class TestHindsightBound(unittest.TestCase):
    def test_preserves_label_range_of_tied_reference_minimizers(self):
        d = [8] * 16
        correct = [100] * 16
        d[1], d[2], d[4], d[8] = 2, 2, 4, 5
        correct[1], correct[2], correct[4] = 97, 99, 101
        d[3] = 0  # Cheaper error at a different byte budget is ineligible.
        self.assertEqual(best_static_mixed(d, correct, 1), {
            "disagreements": 2, "tied_masks": [1, 2], "correct_min": 97, "correct_max": 99,
        })

    def test_fixed_mask_cannot_change_with_each_trial(self):
        # Two eligible masks each make one mistake on non-overlapping trials.
        # A trial-wise picker might make zero; the best fixed mask still makes one.
        reference = [1, 1]
        d = [2] * 16
        for mask, outcomes in [(1, [2, 1]), (2, [1, 2])]:
            d[mask] = sum(p != r for p, r in zip(outcomes, reference))
        self.assertEqual(best_static_mixed(d, [0] * 16, 1)["disagreements"], 1)

    def test_ineligible_budget_rejected(self):
        with self.assertRaises(ValueError):
            best_static_mixed([0] * 16, [0] * 16, 0)


if __name__ == "__main__":
    unittest.main()
