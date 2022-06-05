from abc import abstractmethod
from itertools import groupby
from unittest import TestCase

import torch
from scipy import stats
from torch import flatten

from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.samplers.torch_sampler import TorchBayesianNetworkSampler


class TorchSamplerTestsBase:
    class TestTorchSampler(TestCase):
        @abstractmethod
        def get_torch_device(self) -> torch.device:
            pass

        def setUp(self):
            self.num_samples = 10000
            self.alpha = 0.001

        def test_single_node(self):
            # Assign
            p_true = torch.tensor([1/5, 4/5], dtype=torch.double)
            node = Node(p_true)

            nodes = [node]
            parents = {
                node: [],
            }
            network = BayesianNetwork(nodes, parents)

            # Act
            sut = TorchBayesianNetworkSampler(network, device=self.get_torch_device())

            samples = sut.sample(self.num_samples, nodes)

            # Assert
            samples = samples.cpu()

            expected = p_true * self.num_samples
            actual0 = (samples == 0).sum()
            actual1 = (samples == 1).sum()

            _, p = stats.chisquare([actual0, actual1], expected)

            self.assertGreater(p, self.alpha)

        def test_three_nodes(self):
            # Assign
            p0_true = torch.tensor([1/5, 4/5], dtype=torch.double)
            p1_true = torch.tensor([[2/3, 1/3], [1/9, 8/9]], dtype=torch.double)
            p2_true = torch.tensor([[[3/4, 1/4], [1/2, 1/2]], [[4/7, 3/7], [3/11, 8/11]]], dtype=torch.double)
            node0 = Node(p0_true)
            node1 = Node(p1_true)
            node2 = Node(p2_true)

            nodes = [node0, node1, node2]
            parents = {
                node0: [],
                node1: [node0],
                node2: [node0, node1],
            }
            network = BayesianNetwork(nodes, parents)

            # Act
            sut = TorchBayesianNetworkSampler(network, device=self.get_torch_device())

            samples = sut.sample(self.num_samples, nodes)

            # Assert
            samples = samples.cpu()

            p_full_true = p0_true[:, None, None] * p1_true[:, :, None] * p2_true
            expected = (p_full_true * self.num_samples).flatten()

            num_nodes = 3
            kernel = 2**torch.tensor(list(reversed(range(num_nodes))), dtype=torch.int).reshape([num_nodes, 1])
            states = samples @ kernel
            states = flatten(states)
            states, _ = states.sort()

            actual = [len(list(group)) for _, group in groupby(states)]

            _, p = stats.chisquare(actual, expected)

            self.assertGreater(p, self.alpha)


class TestTorchSamplerCpu(TorchSamplerTestsBase.TestTorchSampler):
    def get_torch_device(self) -> torch.device:
        return torch.device('cpu')


class TestTorchSamplerCuda(TorchSamplerTestsBase.TestTorchSampler):
    def get_torch_device(self) -> torch.device:
        return torch.device('cuda')

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest('Cuda not available')

        super(TestTorchSamplerCuda, self).setUp()
