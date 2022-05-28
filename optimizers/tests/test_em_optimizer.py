from abc import abstractmethod
from typing import List
from unittest import TestCase

import numpy as np
import torch

from common.statistics import generate_random_probability_matrix
from inference_machines.torch_naive_inference_machine import TorchNaiveInferenceMachine
from model.bayesian_network import BayesianNetwork, Node
from optimizers.em_optimizer import EmOptimizer
from samplers.torch_sampler import TorchBayesianNetworkSampler


class EmOptimizerTestBase:
    class TestEmOptimizer(TestCase):
        @abstractmethod
        def get_torch_device(self) -> torch.device:
            pass

        def _generate_random_network(self) -> (BayesianNetwork, List[Node]):
            cpt1 = torch.tensor(generate_random_probability_matrix((2)), dtype=torch.double, device=self.get_torch_device())
            cpt2 = torch.tensor(generate_random_probability_matrix((2, 3)), dtype=torch.double, device=self.get_torch_device())
            cpt3_1 = torch.tensor(generate_random_probability_matrix((2, 3, 4)), dtype=torch.double, device=self.get_torch_device())
            cpt3_2 = torch.tensor(generate_random_probability_matrix((2, 5)), dtype=torch.double, device=self.get_torch_device())
            cpt3_3 = torch.tensor(generate_random_probability_matrix((3, 6)), dtype=torch.double, device=self.get_torch_device())
            Q1 = Node(cpt1, name='Q1')
            Q2 = Node(cpt2, name='Q2')
            Y1 = Node(cpt3_1, name='Y1')
            Y2 = Node(cpt3_2, name='Y2')
            Y3 = Node(cpt3_3, name='Y3')

            nodes = [Q1, Q2, Y1, Y2, Y3]
            parents = {
                Q1: [],
                Q2: [Q1],
                Y1: [Q1, Q2],
                Y2: [Q1],
                Y3: [Q2]
            }

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
                device=self.get_torch_device())

            num_samples = 10000
            self.data = sampler.sample(num_samples, self.observed_nodes)

        def test_optimize_increase_log_likelihood(self):
            # Assign
            untrained_network, observed_nodes = self._generate_random_network()

            # Act
            log_likelihood = torch.zeros(self.num_iterations, dtype=torch.double)

            def inference_machine_factory(bayesian_network):
                return TorchNaiveInferenceMachine(
                    bayesian_network=bayesian_network,
                    observed_nodes=observed_nodes,
                    device=self.get_torch_device())

            sut = EmOptimizer(untrained_network, inference_machine_factory)

            def callback(ll, i):
                log_likelihood[i] = ll

            sut.optimize(
                self.data,
                self.num_iterations,
                callback)

            # Assert either greater or almost equal
            for iteration in range(1, self.num_iterations):
                diff = log_likelihood[iteration] - log_likelihood[iteration-1]

                if diff > 0:
                    self.assertGreaterEqual(diff, 0)
                else:
                    self.assertAlmostEqual(diff, 0)


class TestEmOptimizerCpu(EmOptimizerTestBase.TestEmOptimizer):
    def get_torch_device(self) -> torch.device:
        return torch.device('cpu')


class TestEmOptimizerCuda(EmOptimizerTestBase.TestEmOptimizer):
    def get_torch_device(self) -> torch.device:
        return torch.device('cuda')

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest('Cuda not available')

        super(TestEmOptimizerCuda, self).setUp()