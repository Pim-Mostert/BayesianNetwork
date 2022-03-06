from itertools import groupby
from unittest import TestCase

import numpy as np
import torch
from scipy import stats
from torch import flatten

from common.utilities import Cfg
from inference_engines.torch_naive_inference_machine import TorchNaiveInferenceMachine
from model.bayesian_network import BayesianNetwork
from model.nodes import CPTNode
from samplers.torch_sampler import TorchSampler


class TestTorchNaiveInferenceMachine(TestCase):
    def setUp(self):
        default_cfg = Cfg()
        default_cfg.device = 'cpu'
        self.default_cfg = default_cfg

    def test_no_evidence(self):
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
        sut = TorchNaiveInferenceMachine(self.default_cfg, network, [node2])

        p0_actual = sut.infer([node0])[0]
        p1_actual = sut.infer([node1])[0]
        p2_actual = sut.infer([node2])[0]

        p0x1_actual = sut.infer([node0, node1])[0]
        p1x2_actual = sut.infer([node1, node2])[0]

        p0x1x2_actual = sut.infer([node0, node1, node2])[0]

        # Assert
        p0_expected = p0_true
        p1_expected = (p0_true[:, None] * p1_true).sum(axis=0)
        p2_expected = (p0_true[:, None, None] * p1_true[:, :, None] * p2_true).sum(axis=(0, 1))

        p0x1_expected = p0_true[:, None] * p1_true
        p1x2_expected = (p0_true[:, None, None] * p1_true[:, :, None] * p2_true).sum(axis=0)

        p0x1x2_expected = p0_true[:, None, None] * p1_true[:, :, None] * p2_true

        self.assertArrayAlmostEqual(p0_actual, p0_expected)
        self.assertArrayAlmostEqual(p1_actual, p1_expected)
        self.assertArrayAlmostEqual(p2_actual, p2_expected)

        self.assertArrayAlmostEqual(p0x1_actual, p0x1_expected)
        self.assertArrayAlmostEqual(p1x2_actual, p1x2_expected)

        self.assertArrayAlmostEqual(p0x1x2_actual, p0x1x2_expected)

    def test_all_observed(self):
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
        sut = TorchNaiveInferenceMachine(self.default_cfg, network, nodes)

        evidence = torch.tensor([[0, 1, 0], [1, 1, 1]])
        num_trial = evidence.shape[0]

        ll_actual = sut.enter_evidence(evidence)

        p_actual = [
            sut.infer([node0]),
            sut.infer([node1]),
            sut.infer([node2])
        ]

        px_actual = [
            sut.infer([node0, node1]),
            sut.infer([node1, node2])
        ]

        pxx_actual = sut.infer([node0, node1, node2])

        # Assert - p_posterior
        for i_trial in range(evidence.shape[0]):
            # Single node
            for i_node in range(num_trial):
                p_expected = torch.zeros((2))
                p_expected[evidence[i_trial][i_node]] = 1

                self.assertArrayAlmostEqual(p_actual[i_node][i_trial], p_expected)

            for i_node_pair in range(2):
                px_expected = torch.zeros((2, 2))
                px_expected[
                    evidence[i_trial][i_node_pair],
                    evidence[i_trial][i_node_pair+1]
                ] = 1

                self.assertArrayAlmostEqual(px_actual[i_node_pair][i_trial], px_expected)

            pxx_expected = torch.zeros((2, 2, 2))
            pxx_expected[
                evidence[i_trial][0],
                evidence[i_trial][1],
                evidence[i_trial][2],
            ] = 1
            self.assertArrayAlmostEqual(pxx_actual[i_trial], pxx_expected)

        # Assert - log-likelihood
        ll = torch.zeros(num_trial, dtype=torch.float64)
        p_full = p0_true[:, None, None] * p1_true[:, :, None] * p2_true

        for i_trial in range(num_trial):
            likelihood = p_full[
                evidence[i_trial][0],
                evidence[i_trial][1],
                evidence[i_trial][2]
            ]

            ll[i_trial] = np.log(likelihood)

        ll_expected = ll.sum()

        self.assertAlmostEqual(float(ll_expected), float(ll_actual))

    def test_single_node_observed(self):
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
        sut = TorchNaiveInferenceMachine(self.default_cfg, network, [node2])

        evidence = torch.tensor([[0], [1]])
        num_trial = evidence.shape[0]

        ll_actual = sut.enter_evidence(evidence)

        p_actual_0 = sut.infer([node0])
        p_actual_1 = sut.infer([node1])

        p_actual_0x1 = sut.infer([node0, node1])
        p_actual_1x2 = sut.infer([node1, node2])

        # Assert - p_posterior
        for i_trial in range(evidence.shape[0]):
            # Node0
            p_expected = p0_true[:, None] * p1_true * p2_true[:, :, evidence[i_trial][0]]
            p_expected = p_expected.sum(axis=1)
            p_expected /= p_expected.sum()

            self.assertArrayAlmostEqual(p_actual_0[i_trial], p_expected)

            # Node1
            p_expected = p0_true[:, None] * p1_true * p2_true[:, :, evidence[i_trial][0]]
            p_expected = p_expected.sum(axis=0)
            p_expected /= p_expected.sum()

            self.assertArrayAlmostEqual(p_actual_1[i_trial], p_expected)

            # Node0 x node1
            px_expected = p0_true[:, None] * p1_true * p2_true[:, :, evidence[i_trial][0]]
            px_expected /= px_expected.sum()

            self.assertArrayAlmostEqual(p_actual_0x1[i_trial], px_expected)

            # Node1 x node2
            px_expected = np.zeros((2, 2, 2), dtype=np.float64)
            px_expected[:, :, evidence[i_trial][0]] = p0_true[:, None] * p1_true * p2_true[:, :, evidence[i_trial][0]]
            px_expected = px_expected.sum(axis=0)
            px_expected /= px_expected.sum()

            self.assertArrayAlmostEqual(p_actual_1x2[i_trial], px_expected)

        # Assert - log-likelihood
        ll = torch.zeros(num_trial, dtype=torch.float64)
        p_full = p0_true[:, None, None] * p1_true[:, :, None] * p2_true

        for i_trial in range(num_trial):
            likelihood = p_full[:, :, evidence[i_trial][0]].sum()

            ll[i_trial] = np.log(likelihood)

        ll_expected = ll.sum()

        self.assertAlmostEqual(float(ll_expected), float(ll_actual))

    def assertArrayAlmostEqual(self, actual, expected):
        for a, e in zip(actual.flatten(), expected.flatten()):
            self.assertAlmostEqual(float(a), float(e))