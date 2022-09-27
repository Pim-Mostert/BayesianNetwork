from typing import List
from unittest import TestCase

import torch


class TestCaseExtended(TestCase):
    def assertArrayAlmostEqual(self, actual: torch.tensor, expected: torch.tensor):
        self.assertEqual(actual.shape, expected.shape, msg="Shapes of tensors don't match")

        for a, e in zip(actual.flatten(), expected.flatten()):
            self.assertAlmostEqual(float(a), float(e))
