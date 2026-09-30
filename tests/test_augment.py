import unittest
import numpy as np
from so101_lab.augment import variants


class AugmentTests(unittest.TestCase):
    def test_zero_noise_and_unselected_channels_preserved(self):
        actions = np.arange(18).reshape(6, 3)
        noisy, filtered = variants(actions, [0, 1], [1, 1, 1], 0, 1, 0)
        np.testing.assert_array_equal(actions, noisy)
        np.testing.assert_array_equal(actions, filtered)
        noisy, filtered = variants(actions, [0, 1], [1, 1, 1], 0.2, 0.5, 0)
        np.testing.assert_array_equal(actions[:, 2], noisy[:, 2])
        np.testing.assert_array_equal(actions[:, 2], filtered[:, 2])

    def test_seed_reproducibility_and_no_future_leakage(self):
        actions = np.arange(20).reshape(10, 2)
        first = variants(actions, [0], [1, 1], 0.2, 0.5, 7)
        again = variants(actions, [0], [1, 1], 0.2, 0.5, 7)
        for a, b in zip(first, again):
            np.testing.assert_array_equal(a, b)
        altered = actions.copy()
        altered[7:] += 1000
        changed = variants(altered, [0], [1, 1], 0.2, 0.5, 7)
        np.testing.assert_array_equal(first[1][:7], changed[1][:7])

    def test_episode_initialization_resets(self):
        actions = np.array([[10., 1.], [0., 1.]])
        _, filtered = variants(actions, [0], [0, 0], 0, 0.5, 0)
        self.assertEqual(filtered[0, 0], 10)
        self.assertEqual(filtered[1, 0], 5)
