from typing import List

import torch

from common.utilities import Cfg
from inference_engines.tests.torch_inference_machine_base_tests import TorchInferenceMachineBaseTests
from inference_engines.torch_sum_product_algorithm_inference_machine import TorchSumProductAlgorithmInferenceMachine
from model.bayesian_network import BayesianNetwork
from model.nodes import Node


class TestTorchSumProductAlgorithmInferenceMachineCpu(TorchInferenceMachineBaseTests.TorchInferenceMachineTestCases):
    def create_inference_machine(self, bayesian_network: BayesianNetwork, observed_nodes: List[Node]):
        cfg = Cfg({'device': 'cpu'})
        return TorchSumProductAlgorithmInferenceMachine(cfg, bayesian_network, observed_nodes)


class TestTorchSumProductAlgorithmInferenceMachineGpu(TorchInferenceMachineBaseTests.TorchInferenceMachineTestCases):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest('Cuda not available')

    def create_inference_machine(self, bayesian_network: BayesianNetwork, observed_nodes: List[Node]):
        cfg = Cfg({'device': 'cuda'})
        return TorchSumProductAlgorithmInferenceMachine(cfg, bayesian_network, observed_nodes)
