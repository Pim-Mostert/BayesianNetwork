from unittest import TestCase

import numpy as np
import torch

from common.statistics import generate_random_probability_matrix
from common.utilities import Cfg
from inference_engines.torch_naive_inference_machine import TorchNaiveInferenceMachine
from model.bayesian_network import BayesianNetwork
from model.nodes import CPTNode
from optimizers.em_optimizer import EmOptimizer
from samplers.torch_sampler import TorchSampler


class TestEmOptimizerCpu(TestCase):
    device = 'cpu'

    def setUp(self):
        self.num_iterations = 10

        # Create true network
        self.network, self.observed_nodes = get_true_network()

        # Create training data
        sampler = TorchSampler(Cfg({'device': 'cpu'}), self.network)

        num_samples = 10000
        self.data = sampler.sample(num_samples, self.observed_nodes).to(self.device)

    def test_optimize_increase_log_likelihood(self):
        # Assign
        node0 = CPTNode(generate_random_probability_matrix((2)))
        node1 = CPTNode(generate_random_probability_matrix((2, 2)))
        node2_1 = CPTNode(generate_random_probability_matrix((2, 2, 2)))
        node2_2 = CPTNode(generate_random_probability_matrix((2, 2, 2)))
        node2_3 = CPTNode(generate_random_probability_matrix((2, 2, 2)))
        node2_4 = CPTNode(generate_random_probability_matrix((2, 2, 2)))

        nodes = [node0, node1, node2_1, node2_2, node2_3, node2_4]
        parents = {
            node0: [],
            node1: [node0],
            node2_1: [node0, node1],
            node2_2: [node0, node1],
            node2_3: [node0, node1],
            node2_4: [node0, node1],
        }

        observed_nodes = [node2_1, node2_2, node2_3, node2_4]
        network = BayesianNetwork(nodes, parents)

        # Act
        log_likelihood = np.zeros(self.num_iterations, dtype=np.float64)

        def inference_machine_factory(bayesian_network):
            return TorchNaiveInferenceMachine(
                Cfg({'device': self.device}),
                bayesian_network,
                observed_nodes)

        sut = EmOptimizer(network, inference_machine_factory)

        def callback(ll, i):
            log_likelihood[i] = ll

        sut.optimize(
            self.data,
            self.num_iterations,
            callback)

        # Assert
        for iteration in range(1, self.num_iterations):
            diff = log_likelihood[iteration] - log_likelihood[iteration-1]

            if diff > 0:
                self.assertGreaterEqual(diff, 0)
            else:
                self.assertAlmostEqual(diff, 0)

    def test_train_true_network_no_change(self):
        # Assign

        # Act
        log_likelihood = np.zeros(self.num_iterations, dtype=np.float64)

        def inference_machine_factory(bayesian_network):
            return TorchNaiveInferenceMachine(
                Cfg({'device': self.device}),
                bayesian_network,
                self.observed_nodes)

        sut = EmOptimizer(self.network, inference_machine_factory)

        def callback(ll, i):
            log_likelihood[i] = ll

        sut.optimize(
            self.data,
            self.num_iterations,
            callback)

        # Assert
        # Only first iteration may lead to improvement
        self.assertGreaterEqual(log_likelihood[1], log_likelihood[0])

        # After that, no improvement
        for iteration in range(2, self.num_iterations):
            self.assertAlmostEqual(log_likelihood[iteration], log_likelihood[iteration-1])


def get_true_network():
    node0_1 = CPTNode(np.array([1/5, 4/5], dtype=np.float64))
    node0_2 = CPTNode(np.array([[0.2, 0.8], [0.3, 0.7]], dtype=np.float64))
    node0_3_1 = CPTNode(np.array([[[0, 1], [1, 0]], [[1, 0], [1, 0]]], dtype=np.float64))
    node0_3_2 = CPTNode(np.array([[[1, 0], [0, 1]], [[1, 0], [1, 0]]], dtype=np.float64))
    node0_3_3 = CPTNode(np.array([[[1, 0], [1, 0]], [[0, 1], [1, 0]]], dtype=np.float64))
    node0_3_4 = CPTNode(np.array([[[1, 0], [1, 0]], [[1, 0], [0, 1]]], dtype=np.float64))

    nodes0 = [node0_1, node0_2, node0_3_1, node0_3_2, node0_3_3, node0_3_4]
    parents0 = {
        node0_1: [],
        node0_2: [node0_1],
        node0_3_1: [node0_1, node0_2],
        node0_3_2: [node0_1, node0_2],
        node0_3_3: [node0_1, node0_2],
        node0_3_4: [node0_1, node0_2],
    }

    observed_nodes = [node0_3_1, node0_3_2, node0_3_3, node0_3_4]
    return BayesianNetwork(nodes0, parents0), observed_nodes


class TestEmOptimizerGpu(TestEmOptimizerCpu):
    device = 'cuda'

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest('Cuda not available')

        super(TestEmOptimizerGpu, self).setUp()