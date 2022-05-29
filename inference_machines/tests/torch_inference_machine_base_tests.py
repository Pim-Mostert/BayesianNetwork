from abc import ABC, abstractmethod
from typing import List

import torch

from common.statistics import generate_random_probability_matrix
from common.testcase_extensions import TestCaseExtended
from model.bayesian_network import BayesianNetwork, Node
from model.interfaces import IInferenceMachine


class TorchInferenceMachineBaseTests:
    class NetworkWithSingleParents(TestCaseExtended, ABC):
        @abstractmethod
        def create_inference_machine(self,
                                     bayesian_network: BayesianNetwork,
                                     observed_nodes: List[Node],
                                     num_observations: int) -> IInferenceMachine:
            pass

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.Q1 = Node(
                torch.tensor([1/5, 4/5], dtype=torch.double),
                name='Q1')
            self.Q2 = Node(
                torch.tensor([[2/3, 1/3], [1/9, 8/9]], dtype=torch.double),
                name='Q2')
            self.Y = Node(
                torch.tensor([[4/7, 3/7], [3/11, 8/11]], dtype=torch.double),
                name='Y')

            nodes = [self.Q1, self.Q2, self.Y]
            parents = {
                self.Q1: [],
                self.Q2: [self.Q1],
                self.Y: [self.Q2],
            }
            self.network = BayesianNetwork(nodes, parents)

        def test_no_observations_single_nodes(self):
            # Assign
            p_Q1_expected = torch.einsum('i->i', self.Q1.cpt)[None, ...]
            p_Q2_expected = torch.einsum('i, ij->j', self.Q1.cpt, self.Q2.cpt)[None, ...]
            p_Y_expected = torch.einsum('i, ij, jk->k', self.Q1.cpt, self.Q2.cpt, self.Y.cpt)[None, ...]

            # Act
            sut = self.create_inference_machine(
                bayesian_network=self.network,
                observed_nodes=[self.Q2],
                num_observations=0)

            [p_Q1_actual, p_Q2_actual, p_Y_actual] = sut.infer_single_nodes([self.Q1, self.Q2, self.Y])

            # Assert
            self.assertArrayAlmostEqual(p_Q1_actual, p_Q1_expected)
            self.assertArrayAlmostEqual(p_Q2_actual, p_Q2_expected)
            self.assertArrayAlmostEqual(p_Y_actual, p_Y_expected)

        def test_no_observations_nodes_with_parents(self):
            # Assign
            p_Q1xQ2_expected = torch.einsum('i, ij->ij', self.Q1.cpt, self.Q2.cpt)[None, ...]
            p_Q2xY_expected = torch.einsum('i, ij, jk->jk', self.Q1.cpt, self.Q2.cpt, self.Y.cpt)[None, ...]

            # Act
            sut = self.create_inference_machine(
                bayesian_network=self.network,
                observed_nodes=[self.Q2],
                num_observations=0)

            [p_Q1xQ2_actual, p_Q2xY_actual] = sut.infer_nodes_with_parents([self.Q2, self.Y])

            # Assert
            self.assertArrayAlmostEqual(p_Q1xQ2_actual, p_Q1xQ2_expected)
            self.assertArrayAlmostEqual(p_Q2xY_actual, p_Q2xY_expected)

        def test_all_observed_single_nodes(self):
            # Assign
            evidence = torch.tensor([[0, 0, 0], [0, 1, 1]], dtype=torch.int)
            num_observations = evidence.shape[0]

            evidence_Q1 = torch.zeros((num_observations, 2), dtype=torch.double)
            evidence_Q2 = torch.zeros((num_observations, 2), dtype=torch.double)
            evidence_Y = torch.zeros((num_observations, 2), dtype=torch.double)

            for i_observations in range(num_observations):
                evidence_Q1[i_observations, evidence[i_observations, 0]] = 1
                evidence_Q2[i_observations, evidence[i_observations, 1]] = 1
                evidence_Y[i_observations, evidence[i_observations, 2]] = 1

            p_Q1_expected = torch.einsum('i, ij, jk, ni, nj, nk->ni', self.Q1.cpt, self.Q2.cpt, self.Y.cpt, evidence_Q1, evidence_Q2, evidence_Y)
            p_Q1_expected /= p_Q1_expected.sum(axis=(1), keepdims=True)
            p_Q2_expected = torch.einsum('i, ij, jk, ni, nj, nk->nj', self.Q1.cpt, self.Q2.cpt, self.Y.cpt, evidence_Q1, evidence_Q2, evidence_Y)
            p_Q2_expected /= p_Q2_expected.sum(axis=(1), keepdims=True)
            p_Y_expected = torch.einsum('i, ij, jk, ni, nj, nk->nk', self.Q1.cpt, self.Q2.cpt, self.Y.cpt, evidence_Q1, evidence_Q2, evidence_Y)
            p_Y_expected /= p_Y_expected.sum(axis=(1), keepdims=True)

            # Act
            sut = self.create_inference_machine(
                bayesian_network=self.network,
                observed_nodes=[self.Q1, self.Q2, self.Y],
                num_observations=num_observations)

            sut.enter_evidence(evidence)

            [p_Q1_actual, p_Q2_actual, p_Y_actual] = sut.infer_single_nodes([self.Q1, self.Q2, self.Y])

            # Assert
            self.assertArrayAlmostEqual(p_Q1_actual, p_Q1_expected)
            self.assertArrayAlmostEqual(p_Q2_actual, p_Q2_expected)
            self.assertArrayAlmostEqual(p_Y_actual, p_Y_expected)

        def test_all_observed_log_likelihood(self):
            # Assign
            evidence = torch.tensor([[0, 0, 0], [0, 1, 1]], dtype=torch.int)
            num_observations = evidence.shape[0]

            evidence_Q1 = torch.zeros((num_observations, 2), dtype=torch.double)
            evidence_Q2 = torch.zeros((num_observations, 2), dtype=torch.double)
            evidence_Y = torch.zeros((num_observations, 2), dtype=torch.double)

            for i_observations in range(num_observations):
                evidence_Q1[i_observations, evidence[i_observations, 0]] = 1
                evidence_Q2[i_observations, evidence[i_observations, 1]] = 1
                evidence_Y[i_observations, evidence[i_observations, 2]] = 1

            c = torch.einsum('i, ij, jk, ni, nj, nk->nijk', self.Q1.cpt, self.Q2.cpt, self.Y.cpt, evidence_Q1, evidence_Q2, evidence_Y) \
                .sum(axis=(1, 2, 3))
            ll_expected = torch.log(c).sum()

            # Act
            sut = self.create_inference_machine(
                bayesian_network=self.network,
                observed_nodes=[self.Q1, self.Q2, self.Y],
                num_observations=num_observations)

            sut.enter_evidence(evidence)

            ll_actual = sut.log_likelihood()

            # Assert
            self.assertArrayAlmostEqual(ll_actual, ll_expected)

        def test_all_observed_nodes_with_parents(self):
            # Assign
            evidence = torch.tensor([[0, 0, 0], [0, 1, 1]], dtype=torch.int)
            num_observations = evidence.shape[0]

            evidence_Q1 = torch.zeros((num_observations, 2), dtype=torch.double)
            evidence_Q2 = torch.zeros((num_observations, 2), dtype=torch.double)
            evidence_Y = torch.zeros((num_observations, 2), dtype=torch.double)

            for i_observations in range(num_observations):
                evidence_Q1[i_observations, evidence[i_observations, 0]] = 1
                evidence_Q2[i_observations, evidence[i_observations, 1]] = 1
                evidence_Y[i_observations, evidence[i_observations, 2]] = 1

            p_Q1xQ2_expected = torch.einsum('i, ij, jk, ni, nj, nk->nij', self.Q1.cpt, self.Q2.cpt, self.Y.cpt, evidence_Q1, evidence_Q2, evidence_Y)
            p_Q1xQ2_expected /= p_Q1xQ2_expected.sum(axis=(1, 2), keepdims=True)
            p_Q2xY_expected = torch.einsum('i, ij, jk, ni, nj, nk->njk', self.Q1.cpt, self.Q2.cpt, self.Y.cpt, evidence_Q1, evidence_Q2, evidence_Y)
            p_Q2xY_expected /= p_Q2xY_expected.sum(axis=(1, 2), keepdims=True)

            # Act
            sut = self.create_inference_machine(
                bayesian_network=self.network,
                observed_nodes=[self.Q1, self.Q2, self.Y],
                num_observations=num_observations)

            sut.enter_evidence(evidence)

            [p_Q1xQ2_actual, p_Q2xY_actual] = sut.infer_nodes_with_parents([self.Q2, self.Y])

            # Assert
            self.assertArrayAlmostEqual(p_Q1xQ2_actual, p_Q1xQ2_expected)
            self.assertArrayAlmostEqual(p_Q2xY_actual, p_Q2xY_expected)

        def test_single_node_observed_single_nodes(self):
            # Assign
            evidence = torch.tensor([[0], [1]], dtype=torch.int)
            num_observations = evidence.shape[0]

            evidence_Y = torch.zeros((num_observations, 2), dtype=torch.double)

            for i_observation in range(num_observations):
                evidence_Y[i_observation, evidence[i_observation, 0]] = 1

            p_Q1_expected = torch.einsum('i, ij, jk, nk->ni', self.Q1.cpt, self.Q2.cpt, self.Y.cpt, evidence_Y)
            p_Q1_expected /= p_Q1_expected.sum(axis=(1), keepdims=True)
            p_Q2_expected = torch.einsum('i, ij, jk, nk->nj', self.Q1.cpt, self.Q2.cpt, self.Y.cpt, evidence_Y)
            p_Q2_expected /= p_Q2_expected.sum(axis=(1), keepdims=True)
            p_Y_expected = torch.einsum('i, ij, jk, nk->nk', self.Q1.cpt, self.Q2.cpt, self.Y.cpt, evidence_Y)
            p_Y_expected /= p_Y_expected.sum(axis=(1), keepdims=True)

            # Act
            sut = self.create_inference_machine(
                bayesian_network=self.network,
                observed_nodes=[self.Y],
                num_observations=num_observations)

            sut.enter_evidence(evidence)

            [p_Q1_actual, p_Q2_actual, p_Y_actual] = sut.infer_single_nodes([self.Q1, self.Q2, self.Y])

            # Assert
            self.assertArrayAlmostEqual(p_Q1_actual, p_Q1_expected)
            self.assertArrayAlmostEqual(p_Q2_actual, p_Q2_expected)
            self.assertArrayAlmostEqual(p_Y_actual, p_Y_expected)

        def test_single_node_observed_log_likelihood(self):
            # Assign
            evidence = torch.tensor([[0], [0], [0], [0], [1], [1], [1]], dtype=torch.int)
            num_observations = evidence.shape[0]

            evidence_Y = torch.zeros((num_observations, 2), dtype=torch.double)

            for i_observation in range(num_observations):
                evidence_Y[i_observation, evidence[i_observation, 0]] = 1

            c = torch.einsum('i, ij, jk, nk->nijk', self.Q1.cpt, self.Q2.cpt, self.Y.cpt, evidence_Y) \
                .sum(axis=(1, 2, 3))
            ll_expected = torch.log(c).sum()

            # Act
            sut = self.create_inference_machine(
                bayesian_network=self.network,
                observed_nodes=[self.Y],
                num_observations=num_observations)

            sut.enter_evidence(evidence)

            ll_actual = sut.log_likelihood()

            # Assert
            self.assertArrayAlmostEqual(ll_actual, ll_expected)

        def test_single_node_observed_with_parents(self):
            # Assign
            evidence = torch.tensor([[0], [1]], dtype=torch.int)
            num_observations = evidence.shape[0]

            evidence_Y = torch.zeros((num_observations, 2), dtype=torch.double)

            for i_observations in range(num_observations):
                evidence_Y[i_observations, evidence[i_observations, 0]] = 1

            p_Q1xQ2_expected = torch.einsum('i, ij, jk, nk->nij', self.Q1.cpt, self.Q2.cpt, self.Y.cpt, evidence_Y)
            p_Q1xQ2_expected /= p_Q1xQ2_expected.sum(axis=(1, 2), keepdims=True)
            p_Q2xY_expected = torch.einsum('i, ij, jk, nk->njk', self.Q1.cpt, self.Q2.cpt, self.Y.cpt, evidence_Y)
            p_Q2xY_expected /= p_Q2xY_expected.sum(axis=(1, 2), keepdims=True)

            # Act
            sut = self.create_inference_machine(
                bayesian_network=self.network,
                observed_nodes=[self.Y],
                num_observations=num_observations)

            sut.enter_evidence(evidence)

            [p_Q1xQ2_actual, p_Q2xY_actual] = sut.infer_nodes_with_parents([self.Q2, self.Y])

            # Assert
            self.assertArrayAlmostEqual(p_Q1xQ2_actual, p_Q1xQ2_expected)
            self.assertArrayAlmostEqual(p_Q2xY_actual, p_Q2xY_expected)

    class ComplexNetworkWithSingleParents(TestCaseExtended, ABC):
        @abstractmethod
        def create_inference_machine(self,
                                     bayesian_network: BayesianNetwork,
                                     observed_nodes: List[Node],
                                     num_observations: int) -> IInferenceMachine:
            pass

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.Q1 = Node(
                torch.tensor(generate_random_probability_matrix((2)), dtype=torch.double),
                name='Q1')
            self.Q2 = Node(
                torch.tensor(generate_random_probability_matrix((2, 3)), dtype=torch.double),
                name='Q2')
            self.Q3 = Node(
                torch.tensor(generate_random_probability_matrix((3, 2)), dtype=torch.double),
                name='Q3')
            self.Y1 = Node(
                torch.tensor(generate_random_probability_matrix((2, 2)), dtype=torch.double),
                name='Y1')
            self.Y2 = Node(
                torch.tensor(generate_random_probability_matrix((3, 3)), dtype=torch.double),
                name='Y2')
            self.Y3 = Node(
                torch.tensor(generate_random_probability_matrix((3, 4)), dtype=torch.double),
                name='Y3')
            self.Y4 = Node(
                torch.tensor(generate_random_probability_matrix((2, 2)), dtype=torch.double),
                name='Y4')
            self.Y5 = Node(
                torch.tensor(generate_random_probability_matrix((2, 3)), dtype=torch.double),
                name='Y5')

            nodes = [self.Q1, self.Q2, self.Q3, self.Y1, self.Y2, self.Y3, self.Y4, self.Y5]
            parents = {
                self.Q1: [],
                self.Q2: [self.Q1],
                self.Q3: [self.Q2],
                self.Y1: [self.Q1],
                self.Y2: [self.Q2],
                self.Y3: [self.Q2],
                self.Y4: [self.Q3],
                self.Y5: [self.Q3],
            }
            self.network = BayesianNetwork(nodes, parents)

        def test_inference_single_nodes(self):
            # Assign
            evidence = torch.tensor([[1, 2, 3, 1, 2], [0, 0, 0, 0, 0]], dtype=torch.int)
            num_observations = evidence.shape[0]

            evidence_Y1 = torch.zeros((num_observations, 2), dtype=torch.double)
            evidence_Y2 = torch.zeros((num_observations, 3), dtype=torch.double)
            evidence_Y3 = torch.zeros((num_observations, 4), dtype=torch.double)
            evidence_Y4 = torch.zeros((num_observations, 2), dtype=torch.double)
            evidence_Y5 = torch.zeros((num_observations, 3), dtype=torch.double)

            for i_observation in range(num_observations):
                evidence_Y1[i_observation, evidence[i_observation, 0]] = 1
                evidence_Y2[i_observation, evidence[i_observation, 1]] = 1
                evidence_Y3[i_observation, evidence[i_observation, 2]] = 1
                evidence_Y4[i_observation, evidence[i_observation, 3]] = 1
                evidence_Y5[i_observation, evidence[i_observation, 4]] = 1

            p_Q1_expected = torch.einsum('i, ij, jk, ia, jb, jc, kd, ke, na, nb, nc, nd, ne->ni', self.Q1.cpt, self.Q2.cpt, self.Q3.cpt, self.Y1.cpt, self.Y2.cpt, self.Y3.cpt, self.Y4.cpt, self.Y5.cpt, evidence_Y1, evidence_Y2, evidence_Y3, evidence_Y4, evidence_Y5)
            p_Q1_expected /= p_Q1_expected.sum(axis=(1), keepdims=True)
            p_Q2_expected = torch.einsum('i, ij, jk, ia, jb, jc, kd, ke, na, nb, nc, nd, ne->nj', self.Q1.cpt, self.Q2.cpt, self.Q3.cpt, self.Y1.cpt, self.Y2.cpt, self.Y3.cpt, self.Y4.cpt, self.Y5.cpt, evidence_Y1, evidence_Y2, evidence_Y3, evidence_Y4, evidence_Y5)
            p_Q2_expected /= p_Q2_expected.sum(axis=(1), keepdims=True)
            p_Q3_expected = torch.einsum('i, ij, jk, ia, jb, jc, kd, ke, na, nb, nc, nd, ne->nk', self.Q1.cpt, self.Q2.cpt, self.Q3.cpt, self.Y1.cpt, self.Y2.cpt, self.Y3.cpt, self.Y4.cpt, self.Y5.cpt, evidence_Y1, evidence_Y2, evidence_Y3, evidence_Y4, evidence_Y5)
            p_Q3_expected /= p_Q3_expected.sum(axis=(1), keepdims=True)

            # Act
            sut = self.create_inference_machine(
                bayesian_network=self.network,
                observed_nodes=[self.Y1, self.Y2, self.Y3, self.Y4, self.Y5],
                num_observations=num_observations)

            sut.enter_evidence(evidence)

            [p_Q1_actual, p_Q2_actual, p_Q3_actual] = sut.infer_single_nodes([self.Q1, self.Q2, self.Q3])

            # Assert
            self.assertArrayAlmostEqual(p_Q1_actual, p_Q1_expected)
            self.assertArrayAlmostEqual(p_Q2_actual, p_Q2_expected)
            self.assertArrayAlmostEqual(p_Q3_actual, p_Q3_expected)

        def test_log_likelihood(self):
            # Assign
            evidence = torch.tensor([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 2, 0, 1, 2], [1, 0, 3, 0, 1], [1, 2, 3, 1, 2]] , dtype=torch.int)
            num_observations = evidence.shape[0]

            evidence_Y1 = torch.zeros((num_observations, 2), dtype=torch.double)
            evidence_Y2 = torch.zeros((num_observations, 3), dtype=torch.double)
            evidence_Y3 = torch.zeros((num_observations, 4), dtype=torch.double)
            evidence_Y4 = torch.zeros((num_observations, 2), dtype=torch.double)
            evidence_Y5 = torch.zeros((num_observations, 3), dtype=torch.double)

            for i_observation in range(num_observations):
                evidence_Y1[i_observation, evidence[i_observation, 0]] = 1
                evidence_Y2[i_observation, evidence[i_observation, 1]] = 1
                evidence_Y3[i_observation, evidence[i_observation, 2]] = 1
                evidence_Y4[i_observation, evidence[i_observation, 3]] = 1
                evidence_Y5[i_observation, evidence[i_observation, 4]] = 1

            c = torch.einsum('i, ij, jk, ia, jb, jc, kd, ke, na, nb, nc, nd, ne->n', self.Q1.cpt, self.Q2.cpt, self.Q3.cpt, self.Y1.cpt, self.Y2.cpt, self.Y3.cpt, self.Y4.cpt, self.Y5.cpt, evidence_Y1, evidence_Y2, evidence_Y3, evidence_Y4, evidence_Y5)
            ll_expected = torch.log(c).sum()

            # Act
            sut = self.create_inference_machine(
                bayesian_network=self.network,
                observed_nodes=[self.Y1, self.Y2, self.Y3, self.Y4, self.Y5],
                num_observations=num_observations)

            sut.enter_evidence(evidence)

            ll_actual = sut.log_likelihood()

            # Assert
            self.assertArrayAlmostEqual(ll_actual, ll_expected)

        def test_inference_nodes_with_parents(self):
            # Assign
            evidence = torch.tensor([[1, 2, 3, 1, 2], [0, 0, 0, 0, 0]], dtype=torch.int)
            num_observations = evidence.shape[0]

            evidence_Y1 = torch.zeros((num_observations, 2), dtype=torch.double)
            evidence_Y2 = torch.zeros((num_observations, 3), dtype=torch.double)
            evidence_Y3 = torch.zeros((num_observations, 4), dtype=torch.double)
            evidence_Y4 = torch.zeros((num_observations, 2), dtype=torch.double)
            evidence_Y5 = torch.zeros((num_observations, 3), dtype=torch.double)

            for i_observation in range(num_observations):
                evidence_Y1[i_observation, evidence[i_observation, 0]] = 1
                evidence_Y2[i_observation, evidence[i_observation, 1]] = 1
                evidence_Y3[i_observation, evidence[i_observation, 2]] = 1
                evidence_Y4[i_observation, evidence[i_observation, 3]] = 1
                evidence_Y5[i_observation, evidence[i_observation, 4]] = 1

            p_Q1xQ2_expected = torch.einsum('i, ij, jk, ia, jb, jc, kd, ke, na, nb, nc, nd, ne->nij', self.Q1.cpt, self.Q2.cpt, self.Q3.cpt, self.Y1.cpt, self.Y2.cpt, self.Y3.cpt, self.Y4.cpt, self.Y5.cpt, evidence_Y1, evidence_Y2, evidence_Y3, evidence_Y4, evidence_Y5)
            p_Q1xQ2_expected /= p_Q1xQ2_expected.sum(axis=(1, 2), keepdims=True)
            p_Q2xQ3_expected = torch.einsum('i, ij, jk, ia, jb, jc, kd, ke, na, nb, nc, nd, ne->njk', self.Q1.cpt, self.Q2.cpt, self.Q3.cpt, self.Y1.cpt, self.Y2.cpt, self.Y3.cpt, self.Y4.cpt, self.Y5.cpt, evidence_Y1, evidence_Y2, evidence_Y3, evidence_Y4, evidence_Y5)
            p_Q2xQ3_expected /= p_Q2xQ3_expected.sum(axis=(1, 2), keepdims=True)
            p_Q1xY1_expected = torch.einsum('i, ij, jk, ia, jb, jc, kd, ke, na, nb, nc, nd, ne->nia', self.Q1.cpt, self.Q2.cpt, self.Q3.cpt, self.Y1.cpt, self.Y2.cpt, self.Y3.cpt, self.Y4.cpt, self.Y5.cpt, evidence_Y1, evidence_Y2, evidence_Y3, evidence_Y4, evidence_Y5)
            p_Q1xY1_expected /= p_Q1xY1_expected.sum(axis=(1, 2), keepdims=True)
            p_Q2xY2_expected = torch.einsum('i, ij, jk, ia, jb, jc, kd, ke, na, nb, nc, nd, ne->njb', self.Q1.cpt, self.Q2.cpt, self.Q3.cpt, self.Y1.cpt, self.Y2.cpt, self.Y3.cpt, self.Y4.cpt, self.Y5.cpt, evidence_Y1, evidence_Y2, evidence_Y3, evidence_Y4, evidence_Y5)
            p_Q2xY2_expected /= p_Q2xY2_expected.sum(axis=(1, 2), keepdims=True)
            p_Q2xY3_expected = torch.einsum('i, ij, jk, ia, jb, jc, kd, ke, na, nb, nc, nd, ne->njc', self.Q1.cpt, self.Q2.cpt, self.Q3.cpt, self.Y1.cpt, self.Y2.cpt, self.Y3.cpt, self.Y4.cpt, self.Y5.cpt, evidence_Y1, evidence_Y2, evidence_Y3, evidence_Y4, evidence_Y5)
            p_Q2xY3_expected /= p_Q2xY3_expected.sum(axis=(1, 2), keepdims=True)
            p_Q3xY4_expected = torch.einsum('i, ij, jk, ia, jb, jc, kd, ke, na, nb, nc, nd, ne->nkd', self.Q1.cpt, self.Q2.cpt, self.Q3.cpt, self.Y1.cpt, self.Y2.cpt, self.Y3.cpt, self.Y4.cpt, self.Y5.cpt, evidence_Y1, evidence_Y2, evidence_Y3, evidence_Y4, evidence_Y5)
            p_Q3xY4_expected /= p_Q3xY4_expected.sum(axis=(1, 2), keepdims=True)
            p_Q3xY5_expected = torch.einsum('i, ij, jk, ia, jb, jc, kd, ke, na, nb, nc, nd, ne->nke', self.Q1.cpt, self.Q2.cpt, self.Q3.cpt, self.Y1.cpt, self.Y2.cpt, self.Y3.cpt, self.Y4.cpt, self.Y5.cpt, evidence_Y1, evidence_Y2, evidence_Y3, evidence_Y4, evidence_Y5)
            p_Q3xY5_expected /= p_Q3xY5_expected.sum(axis=(1, 2), keepdims=True)

            # Act
            sut = self.create_inference_machine(
                bayesian_network=self.network,
                observed_nodes=[self.Y1, self.Y2, self.Y3, self.Y4, self.Y5],
                num_observations=num_observations)

            sut.enter_evidence(evidence)

            [p_Q1xQ2_actual, p_Q2xQ3_actual, p_Q1xY1_actual, p_Q2xY2_actual, p_Q2xY3_actual, p_Q3xY4_actual, p_Q3xY5_actual] = sut.infer_nodes_with_parents([self.Q2, self.Q3, self.Y1, self.Y2, self.Y3, self.Y4, self.Y5])

            # Assert
            self.assertArrayAlmostEqual(p_Q1xQ2_actual, p_Q1xQ2_expected)
            self.assertArrayAlmostEqual(p_Q2xQ3_actual, p_Q2xQ3_expected)
            self.assertArrayAlmostEqual(p_Q1xY1_actual, p_Q1xY1_expected)
            self.assertArrayAlmostEqual(p_Q2xY2_actual, p_Q2xY2_expected)
            self.assertArrayAlmostEqual(p_Q2xY3_actual, p_Q2xY3_expected)
            self.assertArrayAlmostEqual(p_Q3xY4_actual, p_Q3xY4_expected)
            self.assertArrayAlmostEqual(p_Q3xY5_actual, p_Q3xY5_expected)

    class NetworkWithMultipleParents(TestCaseExtended, ABC):
        @abstractmethod
        def create_inference_machine(self,
                                     bayesian_network: BayesianNetwork,
                                     observed_nodes: List[Node],
                                     num_observations: int) -> IInferenceMachine:
            pass

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.Q1 = Node(
                torch.tensor([1/5, 4/5], dtype=torch.double),
                name='Q1')
            self.Q2 = Node(
                torch.tensor([[2/3, 1/3], [1/9, 8/9]], dtype=torch.double),
                name='Q2')
            self.Y = Node(
                torch.tensor([[[3/4, 1/4], [1/2, 1/2]],
                          [[4/7, 3/7], [3/11, 8/11]]], dtype=torch.double),
                name='Y')

            nodes = [self.Q1, self.Q2, self.Y]
            parents = {
                self.Q1: [],
                self.Q2: [self.Q1],
                self.Y: [self.Q1, self.Q2],
            }
            self.network = BayesianNetwork(nodes, parents)

        def test_no_observations_single_nodes(self):
            # Assign
            p_Q1_expected = torch.einsum('i->i', self.Q1.cpt)[None, ...]
            p_Q2_expected = torch.einsum('i, ij->j', self.Q1.cpt, self.Q2.cpt)[None, ...]
            p_Y_expected = torch.einsum('i, ij, ijk->k', self.Q1.cpt, self.Q2.cpt, self.Y.cpt)[None, ...]

            # Act
            sut = self.create_inference_machine(
                bayesian_network=self.network,
                observed_nodes=[self.Q2],
                num_observations=0)

            [p_Q1_actual, p_Q2_actual, p_Y_actual] = sut.infer_single_nodes([self.Q1, self.Q2, self.Y])

            # Assert
            self.assertArrayAlmostEqual(p_Q1_actual, p_Q1_expected)
            self.assertArrayAlmostEqual(p_Q2_actual, p_Q2_expected)
            self.assertArrayAlmostEqual(p_Y_actual, p_Y_expected)

        def test_no_observations_nodes_with_parents(self):
            # Assign
            p_Q1xQ2_expected = torch.einsum('i, ij-> ij', self.Q1.cpt, self.Q2.cpt)[None, ...]
            p_Q1xQ2xY_expected = torch.einsum('i, ij, ijk-> ijk', self.Q1.cpt, self.Q2.cpt, self.Y.cpt)[None, ...]

            # Act
            sut = self.create_inference_machine(
                bayesian_network=self.network,
                observed_nodes=[self.Q2],
                num_observations=0)

            [p_Q1xQ2_actual, p_Q1xQ2xY_actual] = sut.infer_nodes_with_parents([self.Q2, self.Y])

            # Assert
            self.assertArrayAlmostEqual(p_Q1xQ2_actual, p_Q1xQ2_expected)
            self.assertArrayAlmostEqual(p_Q1xQ2xY_actual, p_Q1xQ2xY_expected)

        def test_all_observed_single_nodes(self):
            # Assign
            evidence = torch.tensor([[0, 0, 0], [0, 1, 1]], dtype=torch.int)
            num_observations = evidence.shape[0]

            evidence_Q1 = torch.zeros((num_observations, 2), dtype=torch.double)
            evidence_Q2 = torch.zeros((num_observations, 2), dtype=torch.double)
            evidence_Y = torch.zeros((num_observations, 2), dtype=torch.double)

            for i_observations in range(num_observations):
                evidence_Q1[i_observations, evidence[i_observations, 0]] = 1
                evidence_Q2[i_observations, evidence[i_observations, 1]] = 1
                evidence_Y[i_observations, evidence[i_observations, 2]] = 1

            p_Q1_expected = torch.einsum('i, ij, ijk, ni, nj, nk->ni', self.Q1.cpt, self.Q2.cpt, self.Y.cpt, evidence_Q1, evidence_Q2, evidence_Y)
            p_Q1_expected /= p_Q1_expected.sum(axis=(1), keepdims=True)
            p_Q2_expected = torch.einsum('i, ij, ijk, ni, nj, nk->nj', self.Q1.cpt, self.Q2.cpt, self.Y.cpt, evidence_Q1, evidence_Q2, evidence_Y)
            p_Q2_expected /= p_Q2_expected.sum(axis=(1), keepdims=True)
            p_Y_expected = torch.einsum('i, ij, ijk, ni, nj, nk->nk', self.Q1.cpt, self.Q2.cpt, self.Y.cpt, evidence_Q1, evidence_Q2, evidence_Y)
            p_Y_expected /= p_Y_expected.sum(axis=(1), keepdims=True)

            # Act
            sut = self.create_inference_machine(
                bayesian_network=self.network,
                observed_nodes=[self.Q1, self.Q2, self.Y],
                num_observations=num_observations)

            sut.enter_evidence(evidence)

            [p_Q1_actual, p_Q2_actual, p_Y_actual] = sut.infer_single_nodes([self.Q1, self.Q2, self.Y])

            # Assert
            self.assertArrayAlmostEqual(p_Q1_actual, p_Q1_expected)
            self.assertArrayAlmostEqual(p_Q2_actual, p_Q2_expected)
            self.assertArrayAlmostEqual(p_Y_actual, p_Y_expected)

        def test_all_observed_log_likelihood(self):
            # Assign
            evidence = torch.tensor([[0, 0, 0], [0, 1, 1]], dtype=torch.int)
            num_observations = evidence.shape[0]

            evidence_Q1 = torch.zeros((num_observations, 2), dtype=torch.double)
            evidence_Q2 = torch.zeros((num_observations, 2), dtype=torch.double)
            evidence_Y = torch.zeros((num_observations, 2), dtype=torch.double)

            for i_observations in range(num_observations):
                evidence_Q1[i_observations, evidence[i_observations, 0]] = 1
                evidence_Q2[i_observations, evidence[i_observations, 1]] = 1
                evidence_Y[i_observations, evidence[i_observations, 2]] = 1

            c = torch.einsum('i, ij, ijk, ni, nj, nk->nijk', self.Q1.cpt, self.Q2.cpt, self.Y.cpt, evidence_Q1, evidence_Q2, evidence_Y) \
                .sum(axis=(1, 2, 3))
            ll_expected = torch.log(c).sum()

            # Act
            sut = self.create_inference_machine(
                bayesian_network=self.network,
                observed_nodes=[self.Q1, self.Q2, self.Y],
                num_observations=num_observations)

            sut.enter_evidence(evidence)

            ll_actual = sut.log_likelihood()

            # Assert
            self.assertArrayAlmostEqual(ll_actual, ll_expected)

        def test_all_observed_nodes_with_parents(self):
            # Assign
            evidence = torch.tensor([[0, 0, 0], [0, 1, 1]], dtype=torch.int)
            num_observations = evidence.shape[0]

            evidence_Q1 = torch.zeros((num_observations, 2), dtype=torch.double)
            evidence_Q2 = torch.zeros((num_observations, 2), dtype=torch.double)
            evidence_Y = torch.zeros((num_observations, 2), dtype=torch.double)

            for i_observations in range(num_observations):
                evidence_Q1[i_observations, evidence[i_observations, 0]] = 1
                evidence_Q2[i_observations, evidence[i_observations, 1]] = 1
                evidence_Y[i_observations, evidence[i_observations, 2]] = 1

            p_Q1xQ2_expected = torch.einsum('i, ij, ijk, ni, nj, nk->nij', self.Q1.cpt, self.Q2.cpt, self.Y.cpt, evidence_Q1, evidence_Q2, evidence_Y)
            p_Q1xQ2_expected /= p_Q1xQ2_expected.sum(axis=(1, 2), keepdims=True)
            p_Q1xQ2xY_expected = torch.einsum('i, ij, ijk, ni, nj, nk->nijk', self.Q1.cpt, self.Q2.cpt, self.Y.cpt, evidence_Q1, evidence_Q2, evidence_Y)
            p_Q1xQ2xY_expected /= p_Q1xQ2xY_expected.sum(axis=(1, 2, 3), keepdims=True)

            # Act
            sut = self.create_inference_machine(
                bayesian_network=self.network,
                observed_nodes=[self.Q1, self.Q2, self.Y],
                num_observations=num_observations)

            sut.enter_evidence(evidence)

            [p_Q1xQ2_actual, p_Q1xQ2xY_actual] = sut.infer_nodes_with_parents([self.Q2, self.Y])

            # Assert
            self.assertArrayAlmostEqual(p_Q1xQ2_actual, p_Q1xQ2_expected)
            self.assertArrayAlmostEqual(p_Q1xQ2xY_actual, p_Q1xQ2xY_expected)

        def test_single_node_observed_single_nodes(self):
            # Assign
            evidence = torch.tensor([[0], [1]], dtype=torch.int)
            num_observations = evidence.shape[0]

            evidence_Y = torch.zeros((num_observations, 2), dtype=torch.double)

            for i_observation in range(num_observations):
                evidence_Y[i_observation, evidence[i_observation, 0]] = 1

            p_Q1_expected = torch.einsum('i, ij, ijk, nk->ni', self.Q1.cpt, self.Q2.cpt, self.Y.cpt, evidence_Y)
            p_Q1_expected /= p_Q1_expected.sum(axis=(1), keepdims=True)
            p_Q2_expected = torch.einsum('i, ij, ijk, nk->nj', self.Q1.cpt, self.Q2.cpt, self.Y.cpt, evidence_Y)
            p_Q2_expected /= p_Q2_expected.sum(axis=(1), keepdims=True)
            p_Y_expected = torch.einsum('i, ij, ijk, nk->nk', self.Q1.cpt, self.Q2.cpt, self.Y.cpt, evidence_Y)
            p_Y_expected /= p_Y_expected.sum(axis=(1), keepdims=True)

            # Act
            sut = self.create_inference_machine(
                bayesian_network=self.network,
                observed_nodes=[self.Y],
                num_observations=num_observations)

            sut.enter_evidence(evidence)

            [p_Q1_actual, p_Q2_actual, p_Y_actual] = sut.infer_single_nodes([self.Q1, self.Q2, self.Y])

            # Assert
            self.assertArrayAlmostEqual(p_Q1_actual, p_Q1_expected)
            self.assertArrayAlmostEqual(p_Q2_actual, p_Q2_expected)
            self.assertArrayAlmostEqual(p_Y_actual, p_Y_expected)

        def test_single_node_observed_with_parents(self):
            # Assign
            evidence = torch.tensor([[0], [1]], dtype=torch.int)
            num_observations = evidence.shape[0]

            evidence_Y = torch.zeros((num_observations, 2), dtype=torch.double)

            for i_observations in range(num_observations):
                evidence_Y[i_observations, evidence[i_observations, 0]] = 1

            p_Q1xQ2_expected = torch.einsum('i, ij, ijk, nk->nij', self.Q1.cpt, self.Q2.cpt, self.Y.cpt, evidence_Y)
            p_Q1xQ2_expected /= p_Q1xQ2_expected.sum(axis=(1, 2), keepdims=True)
            p_Q1xQ2xY_expected = torch.einsum('i, ij, ijk, nk->nijk', self.Q1.cpt, self.Q2.cpt, self.Y.cpt, evidence_Y)
            p_Q1xQ2xY_expected /= p_Q1xQ2xY_expected.sum(axis=(1, 2, 3), keepdims=True)

            # Act
            sut = self.create_inference_machine(
                bayesian_network=self.network,
                observed_nodes=[self.Y],
                num_observations=num_observations)

            sut.enter_evidence(evidence)

            [p_Q1xQ2_actual, p_Q1xQ2xY_actual] = sut.infer_nodes_with_parents([self.Q2, self.Y])

            # Assert
            self.assertArrayAlmostEqual(p_Q1xQ2_actual, p_Q1xQ2_expected)
            self.assertArrayAlmostEqual(p_Q1xQ2xY_actual, p_Q1xQ2xY_expected)

        def test_single_node_observed_log_likelihood(self):
            # Assign
            evidence = torch.tensor([[0], [1]], dtype=torch.int)
            num_observations = evidence.shape[0]

            evidence_Y = torch.zeros((num_observations, 2), dtype=torch.double)

            for i_observations in range(num_observations):
                evidence_Y[i_observations, evidence[i_observations, 0]] = 1

            c = torch.einsum('i, ij, ijk, nk->nijk', self.Q1.cpt, self.Q2.cpt, self.Y.cpt, evidence_Y) \
                .sum(axis=(1, 2, 3))
            ll_expected = torch.log(c).sum()

            # Act
            sut = self.create_inference_machine(
                bayesian_network=self.network,
                observed_nodes=[self.Y],
                num_observations=num_observations)

            sut.enter_evidence(evidence)

            ll_actual = sut.log_likelihood()

            # Assert
            self.assertArrayAlmostEqual(ll_actual, ll_expected)
