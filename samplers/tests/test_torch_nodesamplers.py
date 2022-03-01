from itertools import groupby
from unittest import TestCase

import numpy as np
import torch
from scipy import stats
from torch import flatten

from common.utilities import Cfg
from model.bayesian_network import BayesianNetwork
from model.nodes import CPTNode
from samplers.torch_sampler import TorchSampler, CPTNodeSampler


class TestTorchNodeSamplers(TestCase):
    def setUp(self):
        default_cfg = Cfg()
        default_cfg.device = 'cpu'
        self.default_cfg = default_cfg

        self.num_samples = 10000
        self.alpha = 0.05

    def test_cptnodesampler(self):
        # Assign
        p_true = np.array([2/14, 4/14, 5/14, 2/14, 1/14], dtype=np.float64)
        node = CPTNode(p_true)

        # Act
        sut = CPTNodeSampler(self.default_cfg, node)

        samples = torch.empty(self.num_samples)
        for i in range(self.num_samples):
            samples[i] = sut.sample([])

        # Assert
        expected = p_true * self.num_samples

        samples, _ = samples.sort()
        actual = [len(list(group)) for _, group in groupby(samples)]

        _, p = stats.chisquare(actual, expected)

        self.assertGreater(p, self.alpha)
