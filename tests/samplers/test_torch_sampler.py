from itertools import groupby
from unittest import TestCase

import torch
from scipy import stats
from torch import flatten

from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.common.torch_settings import TorchSettings
from bayesian_network.samplers.torch_sampler import TorchBayesianNetworkSampler


class TestTorchSampler(TestCase):
    def get_torch_settings(self) -> TorchSettings:
        return TorchSettings(
            device="cpu",
            dtype="float64",
        )

    def setUp(self):
        self.num_samples = 10000
        self.alpha = 0.001

    def test_single_node(self):
        # Assign
        device = self.get_torch_settings().device
        dtype = self.get_torch_settings().dtype

        p_true = torch.tensor([1 / 5, 4 / 5], device=device, dtype=dtype)
        Q = Node(p_true, name="Q")

        nodes = [Q]
        parents = {
            Q: [],
        }
        network = BayesianNetwork(nodes, parents)

        # Act
        sut = TorchBayesianNetworkSampler(network, torch_settings=self.get_torch_settings())

        samples = sut.sample(self.num_samples, nodes)

        # Assert
        samples = samples.cpu()

        expected = p_true.cpu() * self.num_samples
        actual0 = (samples == 0).sum()
        actual1 = (samples == 1).sum()

        _, p = stats.chisquare([actual0, actual1], expected)

        self.assertGreater(p, self.alpha)

    def test_three_nodes(self):
        # Assign
        device = self.get_torch_settings().device
        dtype = self.get_torch_settings().dtype

        p0_true = torch.tensor([1 / 5, 4 / 5], device=device, dtype=dtype)
        p1_true = torch.tensor([[2 / 3, 1 / 3], [1 / 9, 8 / 9]], device=device, dtype=dtype)
        p2_true = torch.tensor(
            [[[3 / 4, 1 / 4], [1 / 2, 1 / 2]], [[4 / 7, 3 / 7], [3 / 11, 8 / 11]]],
            device=device,
            dtype=dtype,
        )
        Q1 = Node(p0_true, name="Q1")
        Q2 = Node(p1_true, name="Q2")
        Q3 = Node(p2_true, name="Q3")

        nodes = [Q1, Q2, Q3]
        parents = {
            Q1: [],
            Q2: [Q1],
            Q3: [Q1, Q2],
        }
        network = BayesianNetwork(nodes, parents)

        # Act
        sut = TorchBayesianNetworkSampler(network, torch_settings=self.get_torch_settings())

        samples = sut.sample(self.num_samples, nodes)

        # Assert
        samples = samples.cpu()

        p_full_true = p0_true[:, None, None] * p1_true[:, :, None] * p2_true
        expected = (p_full_true * self.num_samples).flatten().cpu()

        num_nodes = 3
        kernel = 2 ** torch.tensor(list(reversed(range(num_nodes))), dtype=torch.int).reshape(
            [num_nodes, 1]
        )
        states = samples @ kernel
        states = flatten(states)
        states, _ = states.sort()

        actual = [len(list(group)) for _, group in groupby(states)]

        _, p = stats.chisquare(actual, expected)

        self.assertGreater(p, self.alpha)
