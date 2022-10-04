from abc import abstractmethod, ABC
from typing import List

import torch

from bayesian_network.inference_machines.tests.torch_inference_machine_generic_tests import TorchInferenceMachineGenericTests
from bayesian_network.inference_machines.torch_sum_product_algorithm_inference_machine import TorchSumProductAlgorithmInferenceMachine
from bayesian_network.bayesian_network import BayesianNetwork, Node


# Helper classes
class TestTorchSumProductAlgorithmInferenceMachineBase(ABC):
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
            num_iterations=7,
            num_observations=num_observations,
            callback=lambda factor_graph, iteration: None)


class TestTorchSumProductAlgorithmInferenceMachineBaseCpu(TestTorchSumProductAlgorithmInferenceMachineBase, ABC):
    def get_torch_device(self) -> torch.device:
        return torch.device('cpu')


class TestTorchSumProductAlgorithmInferenceMachineBaseCuda(TestTorchSumProductAlgorithmInferenceMachineBase, ABC):
    def get_torch_device(self) -> torch.device:
        return torch.device('cuda')


# Actual tests
# Cpu
class TestNetworkWithSingleParentsCpu(TestTorchSumProductAlgorithmInferenceMachineBaseCpu, TorchInferenceMachineGenericTests.NetworkWithSingleParents): pass
class TestComplexNetworkWithSingleParentsCpu(TestTorchSumProductAlgorithmInferenceMachineBaseCpu, TorchInferenceMachineGenericTests.ComplexNetworkWithSingleParents): pass
class HandleNumericalUnderflowCpu(TestTorchSumProductAlgorithmInferenceMachineBaseCpu, TorchInferenceMachineGenericTests.HandleNumericalUnderflow): pass


# Cuda
class TestNetworkWithSingleParentsCuda(TestTorchSumProductAlgorithmInferenceMachineBaseCuda, TorchInferenceMachineGenericTests.NetworkWithSingleParents): pass
class TestComplexNetworkWithSingleParentsCuda(TestTorchSumProductAlgorithmInferenceMachineBaseCuda, TorchInferenceMachineGenericTests.ComplexNetworkWithSingleParents): pass
class HandleNumericalUnderflowCuda(TestTorchSumProductAlgorithmInferenceMachineBaseCuda, TorchInferenceMachineGenericTests.HandleNumericalUnderflow): pass

