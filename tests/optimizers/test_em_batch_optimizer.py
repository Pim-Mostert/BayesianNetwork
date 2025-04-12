from typing import List, Tuple
from unittest import TestCase

import torch
from torch.nn.functional import one_hot

from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.common.statistics import generate_random_probability_matrix
from bayesian_network.common.torch_settings import TorchSettings
from bayesian_network.inference_machines.common import InferenceMachineSettings
from bayesian_network.inference_machines.evidence import Evidence, EvidenceBatches
from bayesian_network.inference_machines.naive.naive_inference_machine import NaiveInferenceMachine
from bayesian_network.optimizers.common import OptimizerLogger
from bayesian_network.optimizers.em_batch_optimizer import (
    EmBatchOptimizer,
    EmBatchOptimizerSettings,
)
from bayesian_network.samplers.torch_sampler import TorchBayesianNetworkSampler


class TestEmOptimizer(TestCase):
    def get_torch_settings(self) -> TorchSettings:
        torch_settings = TorchSettings()

        device = torch_settings.device

        print(f"Running tests with configuration: {torch_settings}")

        if device == torch.device("cuda") and not torch.cuda.is_available():
            self.fail("Running tests for cuda, but cuda not available.")

        if device == torch.device("mps") and not torch.backends.mps.is_available():
            self.fail("Running tests for mps, but mps not available.")

        return torch_settings

    def _generate_random_network(
        self,
    ) -> Tuple[BayesianNetwork, List[Node]]:
        device = self.get_torch_settings().device
        dtype = self.get_torch_settings().dtype

        cpt1 = generate_random_probability_matrix((2), device=device, dtype=dtype)
        cpt2 = generate_random_probability_matrix((2, 3), device=device, dtype=dtype)
        cpt3_1 = generate_random_probability_matrix((2, 3, 4), device=device, dtype=dtype)
        cpt3_2 = generate_random_probability_matrix((2, 5), device=device, dtype=dtype)
        cpt3_3 = generate_random_probability_matrix((3, 6), device=device, dtype=dtype)
        Q1 = Node(cpt1, name="Q1")
        Q2 = Node(cpt2, name="Q2")
        Y1 = Node(cpt3_1, name="Y1")
        Y2 = Node(cpt3_2, name="Y2")
        Y3 = Node(cpt3_3, name="Y3")

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
        self.em_batch_optimizer_settings = EmBatchOptimizerSettings(
            num_iterations=10, learning_rate=0.01
        )

        # Create true network
        self.true_network, self.observed_nodes = self._generate_random_network()

        # Create training data
        sampler = TorchBayesianNetworkSampler(
            bayesian_network=self.true_network,
            torch_settings=self.get_torch_settings(),
        )

        num_samples = 10000
        data = sampler.sample(num_samples, self.observed_nodes)
        evidence = Evidence(
            [one_hot(node_data.long()) for node_data in data.T],
            self.get_torch_settings(),
        )

        self.evidence_batches = EvidenceBatches(evidence, 100)

    def test_optimize_increase_log_likelihood_true_data(self):
        # Assign
        untrained_network, observed_nodes = self._generate_random_network()

        # Act
        def inference_machine_factory(bayesian_network):
            return NaiveInferenceMachine(
                settings=InferenceMachineSettings(
                    torch_settings=self.get_torch_settings(),
                    average_log_likelihood=False,
                ),
                bayesian_network=bayesian_network,
                observed_nodes=observed_nodes,
            )

        logger = OptimizerLogger()
        sut = EmBatchOptimizer(
            untrained_network,
            inference_machine_factory,
            settings=self.em_batch_optimizer_settings,
            logger=logger,
        )

        sut.optimize(self.evidence_batches)

        # Assert either greater or almost equal

        ### This naturally fails. TODO: evaluate log_likelihood on true data set
        ll = logger.get_loglikelihood()

        for iteration in range(1, self.em_batch_optimizer_settings.num_iterations):
            diff = ll[iteration] - ll[iteration - 1]

            if diff > 0:
                self.assertGreaterEqual(diff, 0)
            else:
                self.assertAlmostEqual(diff, 0)
