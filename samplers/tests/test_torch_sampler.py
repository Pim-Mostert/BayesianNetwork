from itertools import groupby
from unittest import TestCase

import numpy as np
from scipy import stats
from torch import flatten

from common.utilities import Cfg
from model.bayesian_network import BayesianNetwork
from model.nodes import CPTNode
from samplers.torch_sampler import TorchSampler


class TestTorchSampler(TestCase):
    def setUp(self):
        default_cfg = Cfg()
        default_cfg.device = 'cpu'
        self.default_cfg = default_cfg

        self.num_samples = 10000
        self.alpha = 0.05

    def test_single_cptnode(self):
        # Assign
        p_true = np.array([1/5, 4/5], dtype=np.float64)
        node = CPTNode(p_true)

        nodes = [node]
        parents = {
            node: [],
        }
        network = BayesianNetwork(nodes, parents)

        # Act
        sut = TorchSampler(self.default_cfg, network)

        samples = sut.sample(self.num_samples, nodes)

        # Assert
        expected = p_true * self.num_samples
        actual0 = (samples == 0).sum()
        actual1 = (samples == 1).sum()

        _, p = stats.chisquare([actual0, actual1], expected)

        self.assertGreater(p, self.alpha)

    def test_3layer_cptnodes(self):
        # Assign
        p0_true = np.array([1/5, 4/5], dtype=np.float64)
        p1_true = np.array([[2/3, 1/3], [1/9, 8/9]], dtype=np.float64)
        p2_true = np.array([[[3/4, 1/4], [1/2, 1/2]], [[4/7, 3/7], [3/11, 8/11]]], dtype=np.float64)
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

        samples = sut.sample(self.num_samples, nodes)

        # Assert
        p_full_true = p0_true[:, None, None] * p1_true[:, :, None] * p2_true
        expected = (p_full_true * self.num_samples).flatten()

        num_nodes = 3
        kernel = 2**np.array(list(reversed(range(num_nodes)))).reshape([num_nodes, 1])
        states = samples @ kernel
        states = flatten(states)
        states, _ = states.sort()

        actual = [len(list(group)) for _, group in groupby(states)]

        _, p = stats.chisquare(actual, expected)

        self.assertGreater(p, self.alpha)
