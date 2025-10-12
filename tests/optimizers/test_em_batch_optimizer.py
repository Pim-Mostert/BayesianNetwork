from typing import List, Tuple
from unittest import TestCase
from unittest.mock import create_autospec

import torch
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, TensorDataset

from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.common.statistics import generate_random_probability_matrix
from bayesian_network.common.torch_settings import TorchSettings
from bayesian_network.inference_machines.abstractions import IInferenceMachine
from bayesian_network.inference_machines.common import InferenceMachineSettings
from bayesian_network.inference_machines.evidence import Evidence, EvidenceLoader
from bayesian_network.inference_machines.naive.naive_inference_machine import NaiveInferenceMachine
from bayesian_network.optimizers.common import BatchEvaluator
from bayesian_network.optimizers.em_batch_optimizer import (
    EmBatchOptimizer,
    EmBatchOptimizerSettings,
)
from bayesian_network.samplers.torch_sampler import TorchBayesianNetworkSampler


class TestEmOptimizer(TestCase):
    def get_torch_settings(self) -> TorchSettings:
        return TorchSettings(
            device="cpu",
            dtype="float64",
        )

    def _generate_random_network(
        self,
    ) -> Tuple[BayesianNetwork, List[Node]]:
        Q1 = Node.random((2), torch_settings=self.get_torch_settings(), name="Q1")
        Q2 = Node.random((2, 3), torch_settings=self.get_torch_settings(), name="Q2")
        Y1 = Node.random((2, 3, 4), torch_settings=self.get_torch_settings(), name="Y1")
        Y2 = Node.random((2, 5), torch_settings=self.get_torch_settings(), name="Y2")
        Y3 = Node.random((3, 6), torch_settings=self.get_torch_settings(), name="Y3")

        nodes = [Q1, Q2, Y1, Y2, Y3]
        parents = {
            Q1: [],
            Q2: [Q1],
            Y1: [Q1, Q2],
            Y2: [Q1],
            Y3: [Q2],
        }

        observed_nodes = [Y1, Y2, Y3]
        bayesian_network = BayesianNetwork(nodes, parents)

        return bayesian_network, observed_nodes

    def setUp(self):
        pass

    def test_optimize_increase_log_likelihood_full_data(self):
        ### Assign
        true_network, true_observed_nodes = self._generate_random_network()

        # Create training data
        sampler = TorchBayesianNetworkSampler(
            bayesian_network=true_network,
            torch_settings=self.get_torch_settings(),
        )

        num_samples = 1000
        data = sampler.sample(num_samples, true_observed_nodes)

        evidence_loader = EvidenceLoader(
            data_loader=DataLoader(
                dataset=TensorDataset(data, torch.zeros(num_samples)),
                batch_size=100,
            ),
            transform=lambda batch: Evidence(
                [one_hot(x.long()) for x in batch.T],
                self.get_torch_settings(),
            ),
        )

        # Create untrained network
        untrained_network, observed_nodes = self._generate_random_network()

        def inference_machine_factory(network):
            return NaiveInferenceMachine(
                settings=InferenceMachineSettings(
                    torch_settings=self.get_torch_settings(),
                    average_log_likelihood=False,
                ),
                bayesian_network=network,
                observed_nodes=observed_nodes,
            )

        evaluator = BatchEvaluator(
            inference_machine_factory=inference_machine_factory,
            evidence_loader=evidence_loader,
        )

        sut = EmBatchOptimizer(
            settings=EmBatchOptimizerSettings(
                num_epochs=2,
                learning_rate=0.01,
            ),
            bayesian_network=untrained_network,
            inference_machine_factory=inference_machine_factory,
            evaluator=evaluator,
        )

        # Act
        sut.optimize(evidence_loader)

        # Assert either greater or almost equal
        ll = list(evaluator.log_likelihoods.values())

        for iteration in range(1, len(ll)):
            diff = ll[iteration] - ll[iteration - 1]

            if diff > 0:
                self.assertGreaterEqual(diff, 0)
            else:
                self.assertAlmostEqual(diff, 0)

    def test_optimize_em_step(self):
        ### Assign
        # Setup network
        Q1_cpt = generate_random_probability_matrix((2), torch_settings=self.get_torch_settings())
        Q2_cpt = generate_random_probability_matrix(
            (2, 2), torch_settings=self.get_torch_settings()
        )
        Y_cpt = generate_random_probability_matrix((2, 2), torch_settings=self.get_torch_settings())

        Q1 = Node(Q1_cpt, name="Q1")
        Q2 = Node(Q2_cpt, name="Q2")
        Y = Node(Y_cpt, name="Y")

        nodes = [Q1, Q2, Y]
        parents = {
            Q1: [],
            Q2: [Q1],
            Y: [Q2],
        }

        network = BayesianNetwork(nodes, parents)

        # Setup mock
        inference_machine_mock = create_autospec(IInferenceMachine, instance=True)
        p_Q1 = generate_random_probability_matrix(
            (1, *Q1_cpt.size()), torch_settings=self.get_torch_settings()
        )
        p_Q2 = generate_random_probability_matrix(
            (1, *Q2_cpt.size()), torch_settings=self.get_torch_settings()
        )
        inference_machine_mock.infer_nodes_with_parents.return_value = [p_Q1, p_Q2]

        lr = 0.1
        sut = EmBatchOptimizer(
            settings=EmBatchOptimizerSettings(
                learning_rate=lr,
                num_epochs=1,
            ),
            bayesian_network=network,
            inference_machine_factory=lambda _: inference_machine_mock,
        )

        # Setup evidence loader
        evidence_loader_mock = create_autospec(EvidenceLoader, instance=True)
        evidence_loader_mock.__iter__.return_value = [(None, None)]

        ### Act
        sut.optimize(evidence_loader_mock)

        # Assert
        assert Q1.cpt.isclose((1 - lr) * Q1_cpt + lr * p_Q1).all()
        assert Q2.cpt.isclose((1 - lr) * Q2_cpt + lr * p_Q2).all()

    def test_optimize_em_step_regularization(self):
        ### Assign
        # Setup network
        Q1_cpt = generate_random_probability_matrix((2), torch_settings=self.get_torch_settings())
        Q2_cpt = generate_random_probability_matrix(
            (2, 2), torch_settings=self.get_torch_settings()
        )
        Y_cpt = generate_random_probability_matrix((2, 2), torch_settings=self.get_torch_settings())

        Q1 = Node(Q1_cpt, name="Q1")
        Q2 = Node(Q2_cpt, name="Q2")
        Y = Node(Y_cpt, name="Y")

        nodes = [Q1, Q2, Y]
        parents = {
            Q1: [],
            Q2: [Q1],
            Y: [Q2],
        }

        network = BayesianNetwork(nodes, parents)

        # Setup mock
        inference_machine_mock = create_autospec(IInferenceMachine, instance=True)
        p_Q1 = generate_random_probability_matrix(
            (1, *Q1_cpt.size()), torch_settings=self.get_torch_settings()
        )
        p_Q2 = generate_random_probability_matrix(
            (1, *Q2_cpt.size()), torch_settings=self.get_torch_settings()
        )
        inference_machine_mock.infer_nodes_with_parents.return_value = [p_Q1, p_Q2]

        lr = 0.1
        r = 0.2
        sut = EmBatchOptimizer(
            settings=EmBatchOptimizerSettings(learning_rate=lr, num_epochs=1, regularization=r),
            bayesian_network=network,
            inference_machine_factory=lambda _: inference_machine_mock,
        )

        # Setup evidence loader
        evidence_loader_mock = create_autospec(EvidenceLoader, instance=True)
        evidence_loader_mock.__iter__.return_value = [(None, None)]

        ### Act
        sut.optimize(evidence_loader_mock)

        # Assert
        assert Q1.cpt.isclose((1 - lr) * Q1_cpt + lr * ((1 - r) * p_Q1 + r * 0.5)).all()
        assert Q2.cpt.isclose((1 - lr) * Q2_cpt + lr * ((1 - r) * p_Q2 + r * 0.5)).all()
