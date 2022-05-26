from itertools import groupby
from unittest import TestCase

import torch
from scipy import stats
from torch import flatten

from common.utilities import Cfg
from model.bayesian_network import BayesianNetwork
from model.nodes import CPTNode
from samplers.torch_sampler import TorchSampler


class TestTorchSamplerCpu(TestCase):
    device = 'cpu'

    def setUp(self):
        default_cfg = Cfg()
        default_cfg.device = self.device
        self.default_cfg = default_cfg

        self.num_samples = 10000
        self.alpha = 0.001

    def test_single_cptnode(self):
        # Assign
        p_true = torch.tensor([1/5, 4/5], dtype=torch.double)
        node = CPTNode(p_true)

        nodes = [node]
        parents = {
            node: [],
        }
        network = BayesianNetwork(nodes, parents)

        # Act
        sut = TorchSampler(self.default_cfg, network)

        samples = sut.sample(self.num_samples, nodes).cpu()

        # Assert
        expected = p_true * self.num_samples
        actual0 = (samples == 0).sum()
        actual1 = (samples == 1).sum()

        _, p = stats.chisquare([actual0, actual1], expected)

        self.assertGreater(p, self.alpha)

    def test_3layer_cptnodes(self):
        # Assign
        p0_true = torch.tensor([1/5, 4/5], dtype=torch.double)
        p1_true = torch.tensor([[2/3, 1/3], [1/9, 8/9]], dtype=torch.double)
        p2_true = torch.tensor([[[3/4, 1/4], [1/2, 1/2]], [[4/7, 3/7], [3/11, 8/11]]], dtype=torch.double)
        node0 = CPTNode(p0_true)
        node1 = CPTNode(p1_true)
        node2 = CPTNode(p2_true)

        nodes = [node0, node1, node2]
        parents = {
            node0: [],
            node1: [node0],
            node2: [node0, node1],
        }
        network = BayesianNetwork(nodes, parents)

        # Act
        sut = TorchSampler(self.default_cfg, network)

        samples = sut.sample(self.num_samples, nodes).cpu()

        # Assert
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


class TestTorchSamplerGpu(TestTorchSamplerCpu):
    device = 'cuda'

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest('Cuda not available')

        super(TestTorchSamplerGpu, self).setUp()
