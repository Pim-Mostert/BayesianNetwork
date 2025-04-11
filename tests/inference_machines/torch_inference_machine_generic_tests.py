from abc import ABC, abstractmethod
from typing import List

import torch

from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.common.statistics import generate_random_probability_matrix
from bayesian_network.common.tensor_helpers import rescale_tensors
from bayesian_network.common.testcase_extensions import TestCaseExtended
from bayesian_network.common.torch_settings import TorchSettings
from bayesian_network.inference_machines.evidence import Evidence
from bayesian_network.inference_machines.interfaces import IInferenceMachine


class TorchInferenceMachineGenericTests:
    class TorchInferenceMachineGenericTestsBase(TestCaseExtended, ABC):
        def setUp(self):
            device = self.get_torch_settings().device

            print(f"Running tests with configuration: {self.get_torch_settings()}")

            if device == torch.device("cuda") and not torch.cuda.is_available():
                self.fail("Running tests for cuda, but cuda not available.")

            if device == torch.device("mps") and not torch.backends.mps.is_available():
                self.fail("Running tests for mps, but mps not available.")

        @abstractmethod
        def get_torch_settings(self) -> TorchSettings:
            pass

        @abstractmethod
        def create_inference_machine(
            self,
            bayesian_network: BayesianNetwork,
            observed_nodes: List[Node],
            num_observations: int,
        ) -> IInferenceMachine:
            pass

    class NetworkWithSingleParents(TorchInferenceMachineGenericTestsBase, ABC):
        def setUp(self):
            super().setUp()

            device = self.get_torch_settings().device
            dtype = self.get_torch_settings().dtype

            self.Q1 = Node(
                generate_random_probability_matrix((2), device=device, dtype=dtype),
                name="Q1",
            )
            self.Q2 = Node(
                generate_random_probability_matrix((2, 2), device=device, dtype=dtype),
                name="Q2",
            )
            self.Y = Node(
                generate_random_probability_matrix((2, 2), device=device, dtype=dtype),
                name="Y",
            )

            nodes = [self.Q1, self.Q2, self.Y]
            parents = {
                self.Q1: [],
                self.Q2: [self.Q1],
                self.Y: [self.Q2],
            }
            self.network = BayesianNetwork(nodes, parents)

        def test_no_observations_single_nodes(self):
            # Assign
            p_Q1_expected = torch.einsum("i->i", self.Q1.cpt)[None, ...]
            p_Q2_expected = torch.einsum("i, ij->j", self.Q1.cpt, self.Q2.cpt)[None, ...]
            p_Y_expected = torch.einsum("i, ij, jk->k", self.Q1.cpt, self.Q2.cpt, self.Y.cpt)[
                None, ...
            ]

            # Act
            sut = self.create_inference_machine(
                bayesian_network=self.network,
                observed_nodes=[self.Q2],
                num_observations=0,
            )

            [p_Q1_actual, p_Q2_actual, p_Y_actual] = sut.infer_single_nodes(
                [self.Q1, self.Q2, self.Y]
            )

            # Assert
            self.assertArrayAlmostEqual(p_Q1_actual, p_Q1_expected)
            self.assertArrayAlmostEqual(p_Q2_actual, p_Q2_expected)
            self.assertArrayAlmostEqual(p_Y_actual, p_Y_expected)

        def test_no_observations_nodes_with_parents(self):
            # Assign
            p_Q1xQ2_expected = torch.einsum("i, ij->ij", self.Q1.cpt, self.Q2.cpt)[None, ...]
            p_Q2xY_expected = torch.einsum("i, ij, jk->jk", self.Q1.cpt, self.Q2.cpt, self.Y.cpt)[
                None, ...
            ]

            # Act
            sut = self.create_inference_machine(
                bayesian_network=self.network,
                observed_nodes=[self.Q2],
                num_observations=0,
            )

            [p_Q1xQ2_actual, p_Q2xY_actual] = sut.infer_nodes_with_parents([self.Q2, self.Y])

            # Assert
            self.assertArrayAlmostEqual(p_Q1xQ2_actual, p_Q1xQ2_expected)
            self.assertArrayAlmostEqual(p_Q2xY_actual, p_Q2xY_expected)

        def test_all_observed_single_nodes(self):
            # Assign
            device = self.get_torch_settings().device
            dtype = self.get_torch_settings().dtype

            evidence = rescale_tensors(
                [
                    torch.tensor([[1, 0], [1, 0]], device=device, dtype=dtype),
                    torch.tensor([[1, 0], [0, 1]], device=device, dtype=dtype),
                    torch.tensor([[1, 0], [0, 1]], device=device, dtype=dtype),
                ]
            )
            num_observations = evidence[0].shape[0]

            p_Q1_expected = torch.einsum(
                "i, ij, jk, ni, nj, nk->ni",
                self.Q1.cpt,
                self.Q2.cpt,
                self.Y.cpt,
                *evidence,
            )
            p_Q1_expected /= p_Q1_expected.sum(axis=(1), keepdims=True)
            p_Q2_expected = torch.einsum(
                "i, ij, jk, ni, nj, nk->nj",
                self.Q1.cpt,
                self.Q2.cpt,
                self.Y.cpt,
                *evidence,
            )
            p_Q2_expected /= p_Q2_expected.sum(axis=(1), keepdims=True)
            p_Y_expected = torch.einsum(
                "i, ij, jk, ni, nj, nk->nk",
                self.Q1.cpt,
                self.Q2.cpt,
                self.Y.cpt,
                *evidence,
            )
            p_Y_expected /= p_Y_expected.sum(axis=(1), keepdims=True)

            # Act
            sut = self.create_inference_machine(
                bayesian_network=self.network,
                observed_nodes=[self.Q1, self.Q2, self.Y],
                num_observations=num_observations,
            )

            sut.enter_evidence(
                Evidence(
                    evidence,
                    self.get_torch_settings(),
                )
            )

            [p_Q1_actual, p_Q2_actual, p_Y_actual] = sut.infer_single_nodes(
                [self.Q1, self.Q2, self.Y]
            )

            # Assert
            self.assertArrayAlmostEqual(p_Q1_actual, p_Q1_expected)
            self.assertArrayAlmostEqual(p_Q2_actual, p_Q2_expected)
            self.assertArrayAlmostEqual(p_Y_actual, p_Y_expected)

        def test_all_observed_log_likelihood(self):
            # Assign
            device = self.get_torch_settings().device
            dtype = self.get_torch_settings().dtype

            evidence = rescale_tensors(
                [
                    torch.tensor([[1, 0], [1, 0]], device=device, dtype=dtype),
                    torch.tensor([[1, 0], [0, 1]], device=device, dtype=dtype),
                    torch.tensor([[1, 0], [0, 1]], device=device, dtype=dtype),
                ]
            )
            num_observations = evidence[0].shape[0]

            c = torch.einsum(
                "i, ij, jk, ni, nj, nk->nijk",
                self.Q1.cpt,
                self.Q2.cpt,
                self.Y.cpt,
                *evidence,
            ).sum(axis=(1, 2, 3))
            ll_expected = torch.log(c).sum()

            # Act
            sut = self.create_inference_machine(
                bayesian_network=self.network,
                observed_nodes=[self.Q1, self.Q2, self.Y],
                num_observations=num_observations,
            )

            sut.enter_evidence(
                Evidence(
                    evidence,
                    self.get_torch_settings(),
                )
            )

            ll_actual = sut.log_likelihood()

            # Assert
            self.assertArrayAlmostEqual(ll_actual, ll_expected)

        def test_all_observed_nodes_with_parents(self):
            # Assign
            device = self.get_torch_settings().device
            dtype = self.get_torch_settings().dtype

            evidence = rescale_tensors(
                [
                    torch.tensor([[1, 0], [1, 0]], device=device, dtype=dtype),
                    torch.tensor([[1, 0], [0, 1]], device=device, dtype=dtype),
                    torch.tensor([[1, 0], [0, 1]], device=device, dtype=dtype),
                ]
            )
            num_observations = evidence[0].shape[0]

            p_Q1xQ2_expected = torch.einsum(
                "i, ij, jk, ni, nj, nk->nij",
                self.Q1.cpt,
                self.Q2.cpt,
                self.Y.cpt,
                *evidence,
            )
            p_Q1xQ2_expected /= p_Q1xQ2_expected.sum(axis=(1, 2), keepdims=True)
            p_Q2xY_expected = torch.einsum(
                "i, ij, jk, ni, nj, nk->njk",
                self.Q1.cpt,
                self.Q2.cpt,
                self.Y.cpt,
                *evidence,
            )
            p_Q2xY_expected /= p_Q2xY_expected.sum(axis=(1, 2), keepdims=True)

            # Act
            sut = self.create_inference_machine(
                bayesian_network=self.network,
                observed_nodes=[self.Q1, self.Q2, self.Y],
                num_observations=num_observations,
            )

            sut.enter_evidence(
                Evidence(
                    evidence,
                    self.get_torch_settings(),
                )
            )

            [p_Q1xQ2_actual, p_Q2xY_actual] = sut.infer_nodes_with_parents([self.Q2, self.Y])

            # Assert
            self.assertArrayAlmostEqual(p_Q1xQ2_actual, p_Q1xQ2_expected)
            self.assertArrayAlmostEqual(p_Q2xY_actual, p_Q2xY_expected)

        def test_single_node_observed_single_nodes(self):
            # Assign
            device = self.get_torch_settings().device
            dtype = self.get_torch_settings().dtype

            evidence = rescale_tensors(
                [
                    torch.tensor([[0, 1], [1, 0]], device=device, dtype=dtype),
                ]
            )
            num_observations = evidence[0].shape[0]

            p_Q1_expected = torch.einsum(
                "i, ij, jk, nk->ni", self.Q1.cpt, self.Q2.cpt, self.Y.cpt, *evidence
            )
            p_Q1_expected /= p_Q1_expected.sum(axis=(1), keepdims=True)
            p_Q2_expected = torch.einsum(
                "i, ij, jk, nk->nj", self.Q1.cpt, self.Q2.cpt, self.Y.cpt, *evidence
            )
            p_Q2_expected /= p_Q2_expected.sum(axis=(1), keepdims=True)
            p_Y_expected = torch.einsum(
                "i, ij, jk, nk->nk", self.Q1.cpt, self.Q2.cpt, self.Y.cpt, *evidence
            )
            p_Y_expected /= p_Y_expected.sum(axis=(1), keepdims=True)

            # Act
            sut = self.create_inference_machine(
                bayesian_network=self.network,
                observed_nodes=[self.Y],
                num_observations=num_observations,
            )

            sut.enter_evidence(
                Evidence(
                    evidence,
                    self.get_torch_settings(),
                )
            )

            [p_Q1_actual, p_Q2_actual, p_Y_actual] = sut.infer_single_nodes(
                [self.Q1, self.Q2, self.Y]
            )

            # Assert
            self.assertArrayAlmostEqual(p_Q1_actual, p_Q1_expected)
            self.assertArrayAlmostEqual(p_Q2_actual, p_Q2_expected)
            self.assertArrayAlmostEqual(p_Y_actual, p_Y_expected)

        def test_single_node_observed_log_likelihood(self):
            # Assign
            device = self.get_torch_settings().device
            dtype = self.get_torch_settings().dtype

            evidence = rescale_tensors(
                [
                    torch.tensor([[0, 1], [1, 0]], device=device, dtype=dtype),
                ]
            )
            num_observations = evidence[0].shape[0]

            c = torch.einsum(
                "i, ij, jk, nk->nijk", self.Q1.cpt, self.Q2.cpt, self.Y.cpt, *evidence
            ).sum(axis=(1, 2, 3))
            ll_expected = torch.log(c).sum()

            # Act
            sut = self.create_inference_machine(
                bayesian_network=self.network,
                observed_nodes=[self.Y],
                num_observations=num_observations,
            )

            sut.enter_evidence(
                Evidence(
                    evidence,
                    self.get_torch_settings(),
                )
            )

            ll_actual = sut.log_likelihood()

            # Assert
            self.assertArrayAlmostEqual(ll_actual, ll_expected)

        def test_single_node_observed_with_parents(self):
            # Assign
            device = self.get_torch_settings().device
            dtype = self.get_torch_settings().dtype

            evidence = rescale_tensors(
                [
                    torch.tensor([[0, 1], [1, 0]], device=device, dtype=dtype),
                ]
            )
            num_observations = evidence[0].shape[0]

            p_Q1xQ2_expected = torch.einsum(
                "i, ij, jk, nk->nij", self.Q1.cpt, self.Q2.cpt, self.Y.cpt, *evidence
            )
            p_Q1xQ2_expected /= p_Q1xQ2_expected.sum(axis=(1, 2), keepdims=True)
            p_Q2xY_expected = torch.einsum(
                "i, ij, jk, nk->njk", self.Q1.cpt, self.Q2.cpt, self.Y.cpt, *evidence
            )
            p_Q2xY_expected /= p_Q2xY_expected.sum(axis=(1, 2), keepdims=True)

            # Act
            sut = self.create_inference_machine(
                bayesian_network=self.network,
                observed_nodes=[self.Y],
                num_observations=num_observations,
            )

            sut.enter_evidence(
                Evidence(
                    evidence,
                    self.get_torch_settings(),
                )
            )

            [p_Q1xQ2_actual, p_Q2xY_actual] = sut.infer_nodes_with_parents([self.Q2, self.Y])

            # Assert
            self.assertArrayAlmostEqual(p_Q1xQ2_actual, p_Q1xQ2_expected)
            self.assertArrayAlmostEqual(p_Q2xY_actual, p_Q2xY_expected)

    class ComplexNetworkWithSingleParents(TorchInferenceMachineGenericTestsBase, ABC):
        def setUp(self):
            super().setUp()

            device = self.get_torch_settings().device
            dtype = self.get_torch_settings().dtype

            self.Q1 = Node(
                generate_random_probability_matrix((2), device=device, dtype=dtype),
                name="Q1",
            )
            self.Q2 = Node(
                generate_random_probability_matrix((2, 3), device=device, dtype=dtype),
                name="Q2",
            )
            self.Q3 = Node(
                generate_random_probability_matrix((3, 2), device=device, dtype=dtype),
                name="Q3",
            )
            self.Y1 = Node(
                generate_random_probability_matrix((2, 2), device=device, dtype=dtype),
                name="Y1",
            )
            self.Y2 = Node(
                generate_random_probability_matrix((3, 3), device=device, dtype=dtype),
                name="Y2",
            )
            self.Y3 = Node(
                generate_random_probability_matrix((3, 4), device=device, dtype=dtype),
                name="Y3",
            )
            self.Y4 = Node(
                generate_random_probability_matrix((2, 2), device=device, dtype=dtype),
                name="Y4",
            )
            self.Y5 = Node(
                generate_random_probability_matrix((2, 3), device=device, dtype=dtype),
                name="Y5",
            )

            nodes = [
                self.Q1,
                self.Q2,
                self.Q3,
                self.Y1,
                self.Y2,
                self.Y3,
                self.Y4,
                self.Y5,
            ]
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
            device = self.get_torch_settings().device
            dtype = self.get_torch_settings().dtype

            evidence = rescale_tensors(
                [
                    torch.tensor([[0, 1], [1, 0]], device=device, dtype=dtype),
                    torch.tensor([[0, 0, 1], [1, 0, 0]], device=device, dtype=dtype),
                    torch.tensor([[0, 0, 0, 1], [1, 0, 0, 0]], device=device, dtype=dtype),
                    torch.tensor([[0, 1], [1, 0]], device=device, dtype=dtype),
                    torch.tensor([[0, 0, 1], [1, 0, 0]], device=device, dtype=dtype),
                ]
            )
            num_observations = evidence[0].shape[0]

            p_Q1_expected = torch.einsum(
                "i, ij, jk, ia, jb, jc, kd, ke, na, nb, nc, nd, ne->ni",
                self.Q1.cpt,
                self.Q2.cpt,
                self.Q3.cpt,
                self.Y1.cpt,
                self.Y2.cpt,
                self.Y3.cpt,
                self.Y4.cpt,
                self.Y5.cpt,
                *evidence,
            )
            p_Q1_expected /= p_Q1_expected.sum(axis=(1), keepdims=True)
            p_Q2_expected = torch.einsum(
                "i, ij, jk, ia, jb, jc, kd, ke, na, nb, nc, nd, ne->nj",
                self.Q1.cpt,
                self.Q2.cpt,
                self.Q3.cpt,
                self.Y1.cpt,
                self.Y2.cpt,
                self.Y3.cpt,
                self.Y4.cpt,
                self.Y5.cpt,
                *evidence,
            )
            p_Q2_expected /= p_Q2_expected.sum(axis=(1), keepdims=True)
            p_Q3_expected = torch.einsum(
                "i, ij, jk, ia, jb, jc, kd, ke, na, nb, nc, nd, ne->nk",
                self.Q1.cpt,
                self.Q2.cpt,
                self.Q3.cpt,
                self.Y1.cpt,
                self.Y2.cpt,
                self.Y3.cpt,
                self.Y4.cpt,
                self.Y5.cpt,
                *evidence,
            )
            p_Q3_expected /= p_Q3_expected.sum(axis=(1), keepdims=True)

            # Act
            sut = self.create_inference_machine(
                bayesian_network=self.network,
                observed_nodes=[self.Y1, self.Y2, self.Y3, self.Y4, self.Y5],
                num_observations=num_observations,
            )

            sut.enter_evidence(
                Evidence(
                    evidence,
                    self.get_torch_settings(),
                )
            )

            [p_Q1_actual, p_Q2_actual, p_Q3_actual] = sut.infer_single_nodes(
                [self.Q1, self.Q2, self.Q3]
            )

            # Assert
            self.assertArrayAlmostEqual(p_Q1_actual, p_Q1_expected)
            self.assertArrayAlmostEqual(p_Q2_actual, p_Q2_expected)
            self.assertArrayAlmostEqual(p_Q3_actual, p_Q3_expected)

        def test_log_likelihood(self):
            # Assign
            device = self.get_torch_settings().device
            dtype = self.get_torch_settings().dtype

            evidence = rescale_tensors(
                [
                    torch.tensor([[0, 1], [1, 0]], device=device, dtype=dtype),
                    torch.tensor([[0, 0, 1], [1, 0, 0]], device=device, dtype=dtype),
                    torch.tensor([[0, 0, 0, 1], [1, 0, 0, 0]], device=device, dtype=dtype),
                    torch.tensor([[0, 1], [1, 0]], device=device, dtype=dtype),
                    torch.tensor([[0, 0, 1], [1, 0, 0]], device=device, dtype=dtype),
                ]
            )
            num_observations = evidence[0].shape[0]

            c = torch.einsum(
                "i, ij, jk, ia, jb, jc, kd, ke, na, nb, nc, nd, ne->n",
                self.Q1.cpt,
                self.Q2.cpt,
                self.Q3.cpt,
                self.Y1.cpt,
                self.Y2.cpt,
                self.Y3.cpt,
                self.Y4.cpt,
                self.Y5.cpt,
                *evidence,
            )
            ll_expected = torch.log(c).sum()

            # Act
            sut = self.create_inference_machine(
                bayesian_network=self.network,
                observed_nodes=[self.Y1, self.Y2, self.Y3, self.Y4, self.Y5],
                num_observations=num_observations,
            )

            sut.enter_evidence(
                Evidence(
                    evidence,
                    self.get_torch_settings(),
                )
            )

            ll_actual = sut.log_likelihood()

            # Assert
            self.assertArrayAlmostEqual(ll_actual, ll_expected)

        def test_inference_nodes_with_parents(self):
            # Assign
            device = self.get_torch_settings().device
            dtype = self.get_torch_settings().dtype

            evidence = rescale_tensors(
                [
                    torch.tensor([[0, 1], [1, 0]], device=device, dtype=dtype),
                    torch.tensor([[0, 0, 1], [1, 0, 0]], device=device, dtype=dtype),
                    torch.tensor([[0, 0, 0, 1], [1, 0, 0, 0]], device=device, dtype=dtype),
                    torch.tensor([[0, 1], [1, 0]], device=device, dtype=dtype),
                    torch.tensor([[0, 0, 1], [1, 0, 0]], device=device, dtype=dtype),
                ]
            )
            num_observations = evidence[0].shape[0]

            p_Q1xQ2_expected = torch.einsum(
                "i, ij, jk, ia, jb, jc, kd, ke, na, nb, nc, nd, ne->nij",
                self.Q1.cpt,
                self.Q2.cpt,
                self.Q3.cpt,
                self.Y1.cpt,
                self.Y2.cpt,
                self.Y3.cpt,
                self.Y4.cpt,
                self.Y5.cpt,
                *evidence,
            )
            p_Q1xQ2_expected /= p_Q1xQ2_expected.sum(axis=(1, 2), keepdims=True)
            p_Q2xQ3_expected = torch.einsum(
                "i, ij, jk, ia, jb, jc, kd, ke, na, nb, nc, nd, ne->njk",
                self.Q1.cpt,
                self.Q2.cpt,
                self.Q3.cpt,
                self.Y1.cpt,
                self.Y2.cpt,
                self.Y3.cpt,
                self.Y4.cpt,
                self.Y5.cpt,
                *evidence,
            )
            p_Q2xQ3_expected /= p_Q2xQ3_expected.sum(axis=(1, 2), keepdims=True)
            p_Q1xY1_expected = torch.einsum(
                "i, ij, jk, ia, jb, jc, kd, ke, na, nb, nc, nd, ne->nia",
                self.Q1.cpt,
                self.Q2.cpt,
                self.Q3.cpt,
                self.Y1.cpt,
                self.Y2.cpt,
                self.Y3.cpt,
                self.Y4.cpt,
                self.Y5.cpt,
                *evidence,
            )
            p_Q1xY1_expected /= p_Q1xY1_expected.sum(axis=(1, 2), keepdims=True)
            p_Q2xY2_expected = torch.einsum(
                "i, ij, jk, ia, jb, jc, kd, ke, na, nb, nc, nd, ne->njb",
                self.Q1.cpt,
                self.Q2.cpt,
                self.Q3.cpt,
                self.Y1.cpt,
                self.Y2.cpt,
                self.Y3.cpt,
                self.Y4.cpt,
                self.Y5.cpt,
                *evidence,
            )
            p_Q2xY2_expected /= p_Q2xY2_expected.sum(axis=(1, 2), keepdims=True)
            p_Q2xY3_expected = torch.einsum(
                "i, ij, jk, ia, jb, jc, kd, ke, na, nb, nc, nd, ne->njc",
                self.Q1.cpt,
                self.Q2.cpt,
                self.Q3.cpt,
                self.Y1.cpt,
                self.Y2.cpt,
                self.Y3.cpt,
                self.Y4.cpt,
                self.Y5.cpt,
                *evidence,
            )
            p_Q2xY3_expected /= p_Q2xY3_expected.sum(axis=(1, 2), keepdims=True)
            p_Q3xY4_expected = torch.einsum(
                "i, ij, jk, ia, jb, jc, kd, ke, na, nb, nc, nd, ne->nkd",
                self.Q1.cpt,
                self.Q2.cpt,
                self.Q3.cpt,
                self.Y1.cpt,
                self.Y2.cpt,
                self.Y3.cpt,
                self.Y4.cpt,
                self.Y5.cpt,
                *evidence,
            )
            p_Q3xY4_expected /= p_Q3xY4_expected.sum(axis=(1, 2), keepdims=True)
            p_Q3xY5_expected = torch.einsum(
                "i, ij, jk, ia, jb, jc, kd, ke, na, nb, nc, nd, ne->nke",
                self.Q1.cpt,
                self.Q2.cpt,
                self.Q3.cpt,
                self.Y1.cpt,
                self.Y2.cpt,
                self.Y3.cpt,
                self.Y4.cpt,
                self.Y5.cpt,
                *evidence,
            )
            p_Q3xY5_expected /= p_Q3xY5_expected.sum(axis=(1, 2), keepdims=True)

            # Act
            sut = self.create_inference_machine(
                bayesian_network=self.network,
                observed_nodes=[self.Y1, self.Y2, self.Y3, self.Y4, self.Y5],
                num_observations=num_observations,
            )

            sut.enter_evidence(
                Evidence(
                    evidence,
                    self.get_torch_settings(),
                )
            )

            [
                p_Q1xQ2_actual,
                p_Q2xQ3_actual,
                p_Q1xY1_actual,
                p_Q2xY2_actual,
                p_Q2xY3_actual,
                p_Q3xY4_actual,
                p_Q3xY5_actual,
            ] = sut.infer_nodes_with_parents(
                [self.Q2, self.Q3, self.Y1, self.Y2, self.Y3, self.Y4, self.Y5]
            )

            # Assert
            self.assertArrayAlmostEqual(p_Q1xQ2_actual, p_Q1xQ2_expected)
            self.assertArrayAlmostEqual(p_Q2xQ3_actual, p_Q2xQ3_expected)
            self.assertArrayAlmostEqual(p_Q1xY1_actual, p_Q1xY1_expected)
            self.assertArrayAlmostEqual(p_Q2xY2_actual, p_Q2xY2_expected)
            self.assertArrayAlmostEqual(p_Q2xY3_actual, p_Q2xY3_expected)
            self.assertArrayAlmostEqual(p_Q3xY4_actual, p_Q3xY4_expected)
            self.assertArrayAlmostEqual(p_Q3xY5_actual, p_Q3xY5_expected)

    class NetworkWithMultipleParents(TorchInferenceMachineGenericTestsBase, ABC):
        def setUp(self):
            super().setUp()

            device = self.get_torch_settings().device
            dtype = self.get_torch_settings().dtype

            self.Q1 = Node(
                generate_random_probability_matrix((2), device=device, dtype=dtype),
                name="Q1",
            )
            self.Q2 = Node(
                generate_random_probability_matrix((2, 2), device=device, dtype=dtype),
                name="Q2",
            )
            self.Y = Node(
                generate_random_probability_matrix((2, 2, 2), device=device, dtype=dtype),
                name="Y",
            )

            nodes = [self.Q1, self.Q2, self.Y]
            parents = {
                self.Q1: [],
                self.Q2: [self.Q1],
                self.Y: [self.Q1, self.Q2],
            }
            self.network = BayesianNetwork(nodes, parents)

        def test_no_observations_single_nodes(self):
            # Assign
            p_Q1_expected = torch.einsum("i->i", self.Q1.cpt)[None, ...]
            p_Q2_expected = torch.einsum("i, ij->j", self.Q1.cpt, self.Q2.cpt)[None, ...]
            p_Y_expected = torch.einsum("i, ij, ijk->k", self.Q1.cpt, self.Q2.cpt, self.Y.cpt)[
                None, ...
            ]

            # Act
            sut = self.create_inference_machine(
                bayesian_network=self.network,
                observed_nodes=[self.Q2],
                num_observations=0,
            )

            [p_Q1_actual, p_Q2_actual, p_Y_actual] = sut.infer_single_nodes(
                [self.Q1, self.Q2, self.Y]
            )

            # Assert
            self.assertArrayAlmostEqual(p_Q1_actual, p_Q1_expected)
            self.assertArrayAlmostEqual(p_Q2_actual, p_Q2_expected)
            self.assertArrayAlmostEqual(p_Y_actual, p_Y_expected)

        def test_no_observations_nodes_with_parents(self):
            # Assign
            p_Q1xQ2_expected = torch.einsum("i, ij-> ij", self.Q1.cpt, self.Q2.cpt)[None, ...]
            p_Q1xQ2xY_expected = torch.einsum(
                "i, ij, ijk-> ijk", self.Q1.cpt, self.Q2.cpt, self.Y.cpt
            )[None, ...]

            # Act
            sut = self.create_inference_machine(
                bayesian_network=self.network,
                observed_nodes=[self.Q2],
                num_observations=0,
            )

            [p_Q1xQ2_actual, p_Q1xQ2xY_actual] = sut.infer_nodes_with_parents([self.Q2, self.Y])

            # Assert
            self.assertArrayAlmostEqual(p_Q1xQ2_actual, p_Q1xQ2_expected)
            self.assertArrayAlmostEqual(p_Q1xQ2xY_actual, p_Q1xQ2xY_expected)

        def test_all_observed_single_nodes(self):
            # Assign
            device = self.get_torch_settings().device
            dtype = self.get_torch_settings().dtype

            evidence = rescale_tensors(
                [
                    torch.tensor([[1, 0], [1, 0]], device=device, dtype=dtype),
                    torch.tensor([[1, 0], [0, 1]], device=device, dtype=dtype),
                    torch.tensor([[1, 0], [0, 1]], device=device, dtype=dtype),
                ]
            )
            num_observations = evidence[0].shape[0]

            p_Q1_expected = torch.einsum(
                "i, ij, ijk, ni, nj, nk->ni",
                self.Q1.cpt,
                self.Q2.cpt,
                self.Y.cpt,
                *evidence,
            )
            p_Q1_expected /= p_Q1_expected.sum(axis=(1), keepdims=True)
            p_Q2_expected = torch.einsum(
                "i, ij, ijk, ni, nj, nk->nj",
                self.Q1.cpt,
                self.Q2.cpt,
                self.Y.cpt,
                *evidence,
            )
            p_Q2_expected /= p_Q2_expected.sum(axis=(1), keepdims=True)
            p_Y_expected = torch.einsum(
                "i, ij, ijk, ni, nj, nk->nk",
                self.Q1.cpt,
                self.Q2.cpt,
                self.Y.cpt,
                *evidence,
            )
            p_Y_expected /= p_Y_expected.sum(axis=(1), keepdims=True)

            # Act
            sut = self.create_inference_machine(
                bayesian_network=self.network,
                observed_nodes=[self.Q1, self.Q2, self.Y],
                num_observations=num_observations,
            )

            sut.enter_evidence(
                Evidence(
                    evidence,
                    self.get_torch_settings(),
                )
            )

            [p_Q1_actual, p_Q2_actual, p_Y_actual] = sut.infer_single_nodes(
                [self.Q1, self.Q2, self.Y]
            )

            # Assert
            self.assertArrayAlmostEqual(p_Q1_actual, p_Q1_expected)
            self.assertArrayAlmostEqual(p_Q2_actual, p_Q2_expected)
            self.assertArrayAlmostEqual(p_Y_actual, p_Y_expected)

        def test_all_observed_log_likelihood(self):
            # Assign
            device = self.get_torch_settings().device
            dtype = self.get_torch_settings().dtype

            evidence = rescale_tensors(
                [
                    torch.tensor([[1, 0], [1, 0]], device=device, dtype=dtype),
                    torch.tensor([[1, 0], [0, 1]], device=device, dtype=dtype),
                    torch.tensor([[1, 0], [0, 1]], device=device, dtype=dtype),
                ]
            )
            num_observations = evidence[0].shape[0]
            c = torch.einsum(
                "i, ij, ijk, ni, nj, nk->nijk",
                self.Q1.cpt,
                self.Q2.cpt,
                self.Y.cpt,
                *evidence,
            ).sum(axis=(1, 2, 3))
            ll_expected = torch.log(c).sum()

            # Act
            sut = self.create_inference_machine(
                bayesian_network=self.network,
                observed_nodes=[self.Q1, self.Q2, self.Y],
                num_observations=num_observations,
            )

            sut.enter_evidence(
                Evidence(
                    evidence,
                    self.get_torch_settings(),
                )
            )

            ll_actual = sut.log_likelihood()

            # Assert
            self.assertArrayAlmostEqual(ll_actual, ll_expected)

        def test_all_observed_nodes_with_parents(self):
            # Assign
            device = self.get_torch_settings().device
            dtype = self.get_torch_settings().dtype

            evidence = rescale_tensors(
                [
                    torch.tensor([[1, 0], [1, 0]], device=device, dtype=dtype),
                    torch.tensor([[1, 0], [0, 1]], device=device, dtype=dtype),
                    torch.tensor([[1, 0], [0, 1]], device=device, dtype=dtype),
                ]
            )
            num_observations = evidence[0].shape[0]

            p_Q1xQ2_expected = torch.einsum(
                "i, ij, ijk, ni, nj, nk->nij",
                self.Q1.cpt,
                self.Q2.cpt,
                self.Y.cpt,
                *evidence,
            )
            p_Q1xQ2_expected /= p_Q1xQ2_expected.sum(axis=(1, 2), keepdims=True)
            p_Q1xQ2xY_expected = torch.einsum(
                "i, ij, ijk, ni, nj, nk->nijk",
                self.Q1.cpt,
                self.Q2.cpt,
                self.Y.cpt,
                *evidence,
            )
            p_Q1xQ2xY_expected /= p_Q1xQ2xY_expected.sum(axis=(1, 2, 3), keepdims=True)

            # Act
            sut = self.create_inference_machine(
                bayesian_network=self.network,
                observed_nodes=[self.Q1, self.Q2, self.Y],
                num_observations=num_observations,
            )

            sut.enter_evidence(
                Evidence(
                    evidence,
                    self.get_torch_settings(),
                )
            )

            [p_Q1xQ2_actual, p_Q1xQ2xY_actual] = sut.infer_nodes_with_parents([self.Q2, self.Y])

            # Assert
            self.assertArrayAlmostEqual(p_Q1xQ2_actual, p_Q1xQ2_expected)
            self.assertArrayAlmostEqual(p_Q1xQ2xY_actual, p_Q1xQ2xY_expected)

        def test_single_node_observed_single_nodes(self):
            # Assign
            device = self.get_torch_settings().device
            dtype = self.get_torch_settings().dtype

            evidence = rescale_tensors(
                [
                    torch.tensor([[0, 1], [1, 0]], device=device, dtype=dtype),
                ]
            )
            num_observations = evidence[0].shape[0]

            p_Q1_expected = torch.einsum(
                "i, ij, ijk, nk->ni", self.Q1.cpt, self.Q2.cpt, self.Y.cpt, *evidence
            )
            p_Q1_expected /= p_Q1_expected.sum(axis=(1), keepdims=True)
            p_Q2_expected = torch.einsum(
                "i, ij, ijk, nk->nj", self.Q1.cpt, self.Q2.cpt, self.Y.cpt, *evidence
            )
            p_Q2_expected /= p_Q2_expected.sum(axis=(1), keepdims=True)
            p_Y_expected = torch.einsum(
                "i, ij, ijk, nk->nk", self.Q1.cpt, self.Q2.cpt, self.Y.cpt, *evidence
            )
            p_Y_expected /= p_Y_expected.sum(axis=(1), keepdims=True)

            # Act
            sut = self.create_inference_machine(
                bayesian_network=self.network,
                observed_nodes=[self.Y],
                num_observations=num_observations,
            )

            sut.enter_evidence(
                Evidence(
                    evidence,
                    self.get_torch_settings(),
                )
            )

            [p_Q1_actual, p_Q2_actual, p_Y_actual] = sut.infer_single_nodes(
                [self.Q1, self.Q2, self.Y]
            )

            # Assert
            self.assertArrayAlmostEqual(p_Q1_actual, p_Q1_expected)
            self.assertArrayAlmostEqual(p_Q2_actual, p_Q2_expected)
            self.assertArrayAlmostEqual(p_Y_actual, p_Y_expected)

        def test_single_node_observed_with_parents(self):
            # Assign
            device = self.get_torch_settings().device
            dtype = self.get_torch_settings().dtype

            evidence = rescale_tensors(
                [
                    torch.tensor([[0, 1], [1, 0]], device=device, dtype=dtype),
                ]
            )
            num_observations = evidence[0].shape[0]

            p_Q1xQ2_expected = torch.einsum(
                "i, ij, ijk, nk->nij", self.Q1.cpt, self.Q2.cpt, self.Y.cpt, *evidence
            )
            p_Q1xQ2_expected /= p_Q1xQ2_expected.sum(axis=(1, 2), keepdims=True)
            p_Q1xQ2xY_expected = torch.einsum(
                "i, ij, ijk, nk->nijk", self.Q1.cpt, self.Q2.cpt, self.Y.cpt, *evidence
            )
            p_Q1xQ2xY_expected /= p_Q1xQ2xY_expected.sum(axis=(1, 2, 3), keepdims=True)

            # Act
            sut = self.create_inference_machine(
                bayesian_network=self.network,
                observed_nodes=[self.Y],
                num_observations=num_observations,
            )

            sut.enter_evidence(
                Evidence(
                    evidence,
                    self.get_torch_settings(),
                )
            )

            [p_Q1xQ2_actual, p_Q1xQ2xY_actual] = sut.infer_nodes_with_parents([self.Q2, self.Y])

            # Assert
            self.assertArrayAlmostEqual(p_Q1xQ2_actual, p_Q1xQ2_expected)
            self.assertArrayAlmostEqual(p_Q1xQ2xY_actual, p_Q1xQ2xY_expected)

        def test_single_node_observed_log_likelihood(self):
            # Assign
            device = self.get_torch_settings().device
            dtype = self.get_torch_settings().dtype

            evidence = rescale_tensors(
                [
                    torch.tensor([[0, 1], [1, 0]], device=device, dtype=dtype),
                ]
            )
            num_observations = evidence[0].shape[0]

            c = torch.einsum(
                "i, ij, ijk, nk->nijk", self.Q1.cpt, self.Q2.cpt, self.Y.cpt, *evidence
            ).sum(axis=(1, 2, 3))
            ll_expected = torch.log(c).sum()

            # Act
            sut = self.create_inference_machine(
                bayesian_network=self.network,
                observed_nodes=[self.Y],
                num_observations=num_observations,
            )

            sut.enter_evidence(
                Evidence(
                    evidence,
                    self.get_torch_settings(),
                )
            )

            ll_actual = sut.log_likelihood()

            # Assert
            self.assertArrayAlmostEqual(ll_actual, ll_expected)

    class HandleNumericalUnderflow(TorchInferenceMachineGenericTestsBase, ABC):
        def setUp(self):
            super().setUp()

            device = self.get_torch_settings().device
            dtype = self.get_torch_settings().dtype

            self.num_inputs = 10
            self.num_observations = 2

            self.Q = Node(torch.tensor([0.5, 0.5], device=device, dtype=dtype), name="Q")
            self.Ys = [
                Node(
                    torch.tensor(
                        [[1e-100, 1 - 1e-100], [1e-100, 1 - 1e-100]],
                        device=device,
                        dtype=dtype,
                    ),
                    name=f"Y{i}",
                )
                for i in range(self.num_inputs)
            ]

            nodes = [self.Q, *self.Ys]
            parents = {y: [self.Q] for y in self.Ys}
            parents[self.Q] = []

            self.network = BayesianNetwork(nodes, parents)

            self.evidence = Evidence(
                [
                    torch.tensor([[1 - 1e-100, 1e-100]], device=device, dtype=dtype).repeat(
                        (self.num_observations, 1)
                    )
                    for _ in range(self.num_inputs)
                ],
                self.get_torch_settings(),
            )

        def test_inference_single_nodes(self):
            # Assign

            # Act
            sut = self.create_inference_machine(
                bayesian_network=self.network,
                observed_nodes=self.Ys,
                num_observations=self.num_observations,
            )

            sut.enter_evidence(self.evidence)

            [p_Q_actual] = sut.infer_single_nodes([self.Q])
            p_Ys_actual = sut.infer_single_nodes(self.Ys)

            # Assert
            self.assertFalse(p_Q_actual.isnan().any())

            for p_Y_actual in p_Ys_actual:
                self.assertFalse(p_Y_actual.isnan().any())

        def test_log_likelihood(self):
            # Assign

            # Act
            sut = self.create_inference_machine(
                bayesian_network=self.network,
                observed_nodes=self.Ys,
                num_observations=self.num_observations,
            )

            sut.enter_evidence(self.evidence)

            ll_actual = sut.log_likelihood()

            # Assert
            self.assertFalse(ll_actual.isnan())

        def test_inference_nodes_with_parents(self):
            # Assign

            # Act
            sut = self.create_inference_machine(
                bayesian_network=self.network,
                observed_nodes=self.Ys,
                num_observations=self.num_observations,
            )

            sut.enter_evidence(self.evidence)

            p_YxQs_actual = sut.infer_nodes_with_parents(self.Ys)

            # Assert
            for p_YxQ_actual in p_YxQs_actual:
                self.assertFalse(p_YxQ_actual.isnan().any())
