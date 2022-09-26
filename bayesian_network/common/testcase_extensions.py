from typing import List
from unittest import TestCase

import torch


class TestCaseExtended(TestCase):
    def assertArrayAlmostEqual(self, actual: torch.tensor, expected: torch.tensor):
        self.assertEqual(actual.shape, expected.shape, msg="Shapes of tensors don't match")

        for a, e in zip(actual.flatten(), expected.flatten()):
            self.assertAlmostEqual(float(a), float(e))

    def rescale_tensor(self, tensor: torch.Tensor, gamma=0.999999999):
        return tensor*gamma + (1-gamma)/2

    def rescale_tensors(self, tensors: List[torch.Tensor], gamma=0.999999999):
        return [self.rescale_tensor(t, gamma) for t in tensors]