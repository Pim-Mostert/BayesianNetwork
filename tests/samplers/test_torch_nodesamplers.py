from abc import abstractmethod
from itertools import groupby
from unittest import TestCase

import torch
from scipy import stats

from bayesian_network.bayesian_network import Node
from bayesian_network.common.torch_settings import TorchSettings
from bayesian_network.samplers.torch_sampler import NodeSampler


class TorchNodeSamplerTestBase:
    class TestTorchNodeSampler(TestCase):
        @abstractmethod
        def get_torch_settings(self) -> TorchSettings:
            pass

        def setUp(self):
            self.num_samples = 10000
            self.alpha = 0.001

        def test_node_sampler(self):
            device = self.get_torch_settings().device
            dtype = self.get_torch_settings().dtype

            # Assign
            p_true = torch.tensor([2/14, 4/14, 5/14, 2/14, 1/14], device=device, dtype=dtype)
            node = Node(p_true)

            # Act
            sut = NodeSampler(node)

            samples = torch.empty(self.num_samples, device=device, dtype=torch.int32)
            for i in range(self.num_samples):
                samples[i] = sut.sample([])

            # Assert
            p_true_cpu = p_true.cpu().double()
            p_true_cpu /= p_true_cpu.sum()
            expected = p_true_cpu * self.num_samples

            samples, _ = samples.sort()
            actual = [len(list(group)) for _, group in groupby(samples)]

            _, p = stats.chisquare(actual, expected)

            self.assertGreater(p, self.alpha)
            self.assertEqual(5, 6) # try fail in github actions


class TestTorchNodeSamplersCpu(TorchNodeSamplerTestBase.TestTorchNodeSampler):
    def get_torch_settings(self) -> TorchSettings:
        return TorchSettings(torch.device('cpu'), torch.double)


class TestTorchNodeSamplersCuda(TorchNodeSamplerTestBase.TestTorchNodeSampler):
    def get_torch_settings(self) -> TorchSettings:
        return TorchSettings(torch.device('cuda'), torch.double)

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest('Cuda not available')

        super(TestTorchNodeSamplersCuda, self).setUp()


class TestTorchNodeSamplersMps(TorchNodeSamplerTestBase.TestTorchNodeSampler):
    def get_torch_settings(self) -> TorchSettings:
        return TorchSettings(torch.device('mps'), torch.float32)

    def setUp(self):
        if not torch.has_mps:
            self.skipTest('Mps not available')

        super(TestTorchNodeSamplersMps, self).setUp()