from abc import abstractmethod
from itertools import groupby
from unittest import TestCase

import torch
from scipy import stats

from model.bayesian_network import Node
from samplers.torch_sampler import NodeSampler


class TorchNodeSamplerTestBase:
    class TestTorchNodeSamplerCpu(TestCase):
        @abstractmethod
        def get_torch_device(self) -> torch.device:
            pass

        def setUp(self):
            self.num_samples = 10000
            self.alpha = 0.001

        def test_node_sampler(self):
            # Assign
            p_true = torch.tensor([2/14, 4/14, 5/14, 2/14, 1/14], dtype=torch.double)
            node = Node(p_true)

            # Act
            sut = NodeSampler(node, device=self.get_torch_device())

            samples = torch.empty(self.num_samples)
            for i in range(self.num_samples):
                samples[i] = sut.sample([])

            # Assert
            expected = p_true * self.num_samples

            samples, _ = samples.sort()
            actual = [len(list(group)) for _, group in groupby(samples)]

            _, p = stats.chisquare(actual, expected)

            self.assertGreater(p, self.alpha)


class TestTorchNodeSamplersCpi(TorchNodeSamplerTestBase.TestTorchNodeSamplerCpu):
    def get_torch_device(self) -> torch.device:
        return torch.device('cpu')


class TestTorchNodeSamplersCuda(TorchNodeSamplerTestBase.TestTorchNodeSamplerCpu):
    def get_torch_device(self) -> torch.device:
        return torch.device('cuda')

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest('Cuda not available')

        super(TestTorchNodeSamplersCuda, self).setUp()