from unittest import TestCase

import numpy as np
from scipy import stats

from common.utilities import Cfg
from model.bayesian_network import BayesianNetwork
from model.nodes import CPTNode
from samplers.torch_sampler import TorchSampler


class TestTorchSampler(TestCase):
    def setUp(self):
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
        cfg = Cfg()
        cfg.device = 'cpu'

        sut = TorchSampler(cfg, network)

        samples = sut.sample(self.num_samples, nodes)

        # Assert
        expected = p_true * self.num_samples
        actual0 = (samples == 0).sum()
        actual1 = (samples == 1).sum()

        _, p = stats.chisquare([actual0, actual1], expected)

        self.assertGreater(p, self.alpha)

