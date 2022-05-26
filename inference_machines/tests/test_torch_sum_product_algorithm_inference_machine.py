from abc import abstractmethod, ABC
from typing import List

import torch

from inference_machines.tests.torch_inference_machine_base_tests import TorchInferenceMachineBaseTests
from inference_machines.torch_sum_product_algorithm_inference_machine import TorchSumProductAlgorithmInferenceMachine
from model.bayesian_network import BayesianNetwork
from model.nodes import Node


# Helper classes
class TorchSumProductAlgorithmInferenceMachineTestBase:
    class TorchSumProductAlgorithmInferenceMachineTestBase(ABC):
        @abstractmethod
        def get_torch_device(self) -> torch.device:
            pass

        def create_inference_machine(self,
                                     bayesian_network: BayesianNetwork,
                                     observed_nodes: List[Node],
                                     num_observations: int):
            return TorchSumProductAlgorithmInferenceMachine(
                bayesian_network=bayesian_network,
                observed_nodes=observed_nodes,
                device=self.get_torch_device(),
                num_iterations=20,
                num_observations=num_observations,
                callback=lambda factor_graph, iteration: None)

    class TorchSumProductAlgorithmInferenceMachineTestBaseCpu(TorchSumProductAlgorithmInferenceMachineTestBase):
        def get_torch_device(self) -> torch.device:
            return torch.device('cpu')

    class TorchSumProductAlgorithmInferenceMachineTestBaseCuda(TorchSumProductAlgorithmInferenceMachineTestBase):
        def setUp(self):
            if not torch.cuda.is_available():
                self.skipTest('Cuda not available')

        def get_torch_device(self) -> torch.device:
            return torch.device('cuda')


# Run all tests for cpu
class TestNetworkWithSingleParentsCpu(TorchSumProductAlgorithmInferenceMachineTestBase.TorchSumProductAlgorithmInferenceMachineTestBaseCpu,
                                      TorchInferenceMachineBaseTests.NetworkWithSingleParents):
    pass


# Run all tests for cuda
class TestNetworkWithSingleParentsCuda(TorchSumProductAlgorithmInferenceMachineTestBase.TorchSumProductAlgorithmInferenceMachineTestBaseCuda,
                                       TorchInferenceMachineBaseTests.NetworkWithSingleParents):
    pass



