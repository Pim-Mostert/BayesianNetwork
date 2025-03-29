from unittest import TestCase

import torch


class TestCaseExtended(TestCase):
    def assertArrayAlmostEqual(self, actual: torch.tensor, expected: torch.tensor, places=None):
        self.assertEqual(actual.shape, expected.shape, msg="Shapes of tensors don't match")

        if places is None and actual.dtype == torch.float32:
            places = 5

        for a, e in zip(actual.flatten(), expected.flatten()):
            self.assertAlmostEqual(float(a), float(e), places=places)
