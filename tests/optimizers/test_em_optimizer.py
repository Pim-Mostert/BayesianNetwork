from abc import abstractmethod
from typing import List, Tuple
from unittest import TestCase

import torch
from torch.nn.functional import one_hot

from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.common.statistics import generate_random_probability_matrix
from bayesian_network.common.torch_settings import TorchSettings
from bayesian_network.inference_machines.torch_naive_inference_machine import TorchNaiveInferenceMachine
from bayesian_network.optimizers.em_optimizer import EmOptimizer
from bayesian_network.samplers.torch_sampler import TorchBayesianNetworkSampler


class EmOptimizerTestBase:
    class TestEmOptimizer(TestCase):
        @abstractmethod
        def get_torch_settings(self) -> TorchSettings:
            pass

        def _generate_random_network(self) -> Tuple[BayesianNetwork, List[Node]]:
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
            parents = {Q1: [], Q2: [Q1], Y1: [Q1, Q2], Y2: [Q1], Y3: [Q2]}

            observed_nodes = [Y1, Y2, Y3]
            bayesian_network = BayesianNetwork(nodes, parents)

            return bayesian_network, observed_nodes

        def setUp(self):
            self.num_iterations = 10

            # Create true network
            self.true_network, self.observed_nodes = self._generate_random_network()

            # Create training data
            sampler = TorchBayesianNetworkSampler(
                bayesian_network=self.true_network,
                torch_settings=self.get_torch_settings(),
            )

            num_samples = 10000
            data = sampler.sample(num_samples, self.observed_nodes)
            self.data = [one_hot(node_data.long()) for node_data in data.T]

        def test_optimize_increase_log_likelihood(self):
            # Assign
            untrained_network, observed_nodes = self._generate_random_network()

            # Act
            log_likelihood = torch.zeros(self.num_iterations, dtype=torch.double)

            def inference_machine_factory(bayesian_network):
                return TorchNaiveInferenceMachine(
                    bayesian_network=bayesian_network,
                    observed_nodes=observed_nodes,
                    torch_settings=self.get_torch_settings(),
                )

            sut = EmOptimizer(untrained_network, inference_machine_factory)

            def callback(ll, i, duration):
                log_likelihood[i] = ll

            sut.optimize(self.data, self.num_iterations, callback)

            # Assert either greater or almost equal
            for iteration in range(1, self.num_iterations):
                diff = log_likelihood[iteration] - log_likelihood[iteration - 1]

                if diff > 0:
                    self.assertGreaterEqual(diff, 0)
                else:
                    self.assertAlmostEqual(diff, 0)


class TestEmOptimizerCpu(EmOptimizerTestBase.TestEmOptimizer):
    def get_torch_settings(self) -> TorchSettings:
        return TorchSettings(torch.device("cpu"), torch.double)


class TestEmOptimizerCuda(EmOptimizerTestBase.TestEmOptimizer):
    def get_torch_settings(self) -> TorchSettings:
        return TorchSettings(torch.device("cuda"), torch.double)

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("Cuda not available")

        super(TestEmOptimizerCuda, self).setUp()
