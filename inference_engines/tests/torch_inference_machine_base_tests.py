from abc import ABC, abstractmethod
from typing import List
from unittest import TestCase

import numpy as np
import torch

from model.bayesian_network import BayesianNetwork
from model.interfaces import IInferenceMachine
from model.nodes import CPTNode, Node


class TorchInferenceMachineBaseTests:
    class TorchInferenceMachineTestCases(TestCase, ABC):
        @abstractmethod
        def create_inference_machine(self,
                                     bayesian_network: BayesianNetwork,
                                     observed_nodes: List[Node],
                                     num_observations: int) -> IInferenceMachine:
            pass

        def test_no_evidence(self):
            # Assign
            p0_true = np.array([1/5, 4/5], dtype=np.float64)
            p1_true = np.array([[2/3, 1/3], [1/9, 8/9]], dtype=np.float64)
            p2_true = np.array([[[3/4, 1/4], [1/2, 1/2]], [[4/7, 3/7], [3/11, 8/11]]], dtype=np.float64)
            node0 = CPTNode(p0_true, name='node0')
            node1 = CPTNode(p1_true, name='node1')
            node2 = CPTNode(p2_true, name='node2')

            nodes = [node0, node1, node2]
            parents = {
                node0: [],
                node1: [node0],
                node2: [node0, node1],
            }
            network = BayesianNetwork(nodes, parents)

            # Act
            sut = self.create_inference_machine(network, [node2], 0)

            p_actual = sut.infer_single_nodes([node0, node1, node2])
            p0_actual = p_actual[0][0]
            p1_actual = p_actual[1][0]
            p2_actual = p_actual[2][0]

            p_actual = sut.infer_children_with_parents([node1, node2])
            p0x1_actual = p_actual[0][0]
            p0x1x2_actual = p_actual[1][0]

            # Assert
            p0_expected = p0_true
            p1_expected = (p0_true[:, None] * p1_true).sum(axis=0)
            p2_expected = (p0_true[:, None, None] * p1_true[:, :, None] * p2_true).sum(axis=(0, 1))

            p0x1_expected = p0_true[:, None] * p1_true
            p0x1x2_expected = p0_true[:, None, None] * p1_true[:, :, None] * p2_true

            self.assertArrayAlmostEqual(p0_actual, p0_expected)
            self.assertArrayAlmostEqual(p1_actual, p1_expected)
            self.assertArrayAlmostEqual(p2_actual, p2_expected)

            self.assertArrayAlmostEqual(p0x1_actual, p0x1_expected)
            self.assertArrayAlmostEqual(p0x1x2_actual, p0x1x2_expected)

        def test_all_observed(self):
            # Assign
            p0_true = np.array([1/5, 4/5], dtype=np.float64)
            p1_true = np.array([[2/3, 1/3], [1/9, 8/9]], dtype=np.float64)
            p2_true = np.array([[[3/4, 1/4], [1/2, 1/2]], [[4/7, 3/7], [3/11, 8/11]]], dtype=np.float64)
            node0 = CPTNode(p0_true, name='node0')
            node1 = CPTNode(p1_true, name='node1')
            node2 = CPTNode(p2_true, name='node2')

            nodes = [node0, node1, node2]
            parents = {
                node0: [],
                node1: [node0],
                node2: [node0, node1],
            }
            network = BayesianNetwork(nodes, parents)

            # Act
            evidence = torch.tensor([[0, 1, 0], [1, 1, 1]])
            sut = self.create_inference_machine(network, nodes, len(evidence))

            num_trial = evidence.shape[0]

            sut.enter_evidence(evidence)

            p_actual = sut.infer_single_nodes([node0, node1, node2])
            [px_actual, pxx_actual] = sut.infer_children_with_parents([node1, node2])

            # Assert - p_posterior
            for i_trial in range(evidence.shape[0]):
                # Single node
                for i_node in range(num_trial):
                    p_expected = torch.zeros((2))
                    p_expected[evidence[i_trial][i_node]] = 1

                    self.assertArrayAlmostEqual(p_actual[i_node][i_trial], p_expected)

                # Node1 with parent node0
                px_expected = torch.zeros((2, 2))
                px_expected[
                    evidence[i_trial][0],
                    evidence[i_trial][1]
                ] = 1

                self.assertArrayAlmostEqual(px_actual[i_trial], px_expected)

                # Node2 with parents node0 and node1
                pxx_expected = torch.zeros((2, 2, 2))
                pxx_expected[
                    evidence[i_trial][0],
                    evidence[i_trial][1],
                    evidence[i_trial][2],
                ] = 1
                self.assertArrayAlmostEqual(pxx_actual[i_trial], pxx_expected)

            # Assert - log-likelihood
            ll_actual = sut.log_likelihood()

            ll_expected = torch.zeros(num_trial, dtype=torch.float64)
            p_full = p0_true[:, None, None] * p1_true[:, :, None] * p2_true

            for i_trial in range(num_trial):
                likelihood = p_full[
                    evidence[i_trial][0],
                    evidence[i_trial][1],
                    evidence[i_trial][2]
                ]

                ll_expected[i_trial] = np.log(likelihood)

            ll_expected = ll_expected.sum()

            self.assertAlmostEqual(float(ll_expected), float(ll_actual))

        def test_single_node_observed(self):
            # Assign
            p0_true = np.array([1/5, 4/5], dtype=np.float64)
            p1_true = np.array([[2/3, 1/3], [1/9, 8/9]], dtype=np.float64)
            p2_true = np.array([[[3/4, 1/4], [1/2, 1/2]], [[4/7, 3/7], [3/11, 8/11]]], dtype=np.float64)
            node0 = CPTNode(p0_true, name='node0')
            node1 = CPTNode(p1_true, name='node1')
            node2 = CPTNode(p2_true, name='node2')

            nodes = [node0, node1, node2]
            parents = {
                node0: [],
                node1: [node0],
                node2: [node0, node1],
            }
            network = BayesianNetwork(nodes, parents)

            # Act
            evidence = torch.tensor([[0], [1]])
            sut = self.create_inference_machine(network, [node2], len(evidence))

            num_trial = evidence.shape[0]

            sut.enter_evidence(evidence)

            [p_actual_0, p_actual_1] = sut.infer_single_nodes([node0, node1])
            [p_actual_0x1] = sut.infer_children_with_parents([node1])

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

            # Assert - log-likelihood
            ll_actual = sut.log_likelihood()

            ll_expected = torch.zeros(num_trial, dtype=torch.float64)
            p_full = p0_true[:, None, None] * p1_true[:, :, None] * p2_true

            for i_trial in range(num_trial):
                likelihood = p_full[:, :, evidence[i_trial][0]].sum()

                ll_expected[i_trial] = np.log(likelihood)

            ll_expected = ll_expected.sum()

            self.assertAlmostEqual(float(ll_expected), float(ll_actual))

        def assertArrayAlmostEqual(self, actual, expected):
            for a, e in zip(actual.flatten(), expected.flatten()):
                self.assertAlmostEqual(float(a), float(e))
