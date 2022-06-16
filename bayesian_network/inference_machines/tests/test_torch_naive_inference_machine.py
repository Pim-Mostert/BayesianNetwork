from abc import abstractmethod, ABC
from typing import List

import torch

from bayesian_network.inference_machines.tests.torch_inference_machine_generic_tests import TorchInferenceMachineGenericTests
from bayesian_network.inference_machines.torch_naive_inference_machine import TorchNaiveInferenceMachine
from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.interfaces import IInferenceMachine


# Helper classes
class TestTorchNaiveInferenceMachineBase(ABC):
    @abstractmethod
    def get_torch_device(self) -> torch.device:
        pass

    def create_inference_machine(self,
                                 bayesian_network: BayesianNetwork,
                                 observed_nodes: List[Node],
                                 num_observations: int) -> IInferenceMachine:
        return TorchNaiveInferenceMachine(
            bayesian_network=bayesian_network,
            observed_nodes=observed_nodes,
            device=self.get_torch_device())


class TestTorchNaiveInferenceMachineBaseCpu(TestTorchNaiveInferenceMachineBase, ABC):
    def get_torch_device(self) -> torch.device:
        return torch.device('cpu')


class TestTorchNaiveInferenceMachineBaseCuda(TestTorchNaiveInferenceMachineBase, ABC):
    def get_torch_device(self) -> torch.device:
        return torch.device('cuda')


# Actual tests
# Cpu
class TestNetworkWithSingleParentsCpu(TestTorchNaiveInferenceMachineBaseCpu, TorchInferenceMachineGenericTests.NetworkWithSingleParents): pass
class TestComplexNetworkWithSingleParentsCpu(TestTorchNaiveInferenceMachineBaseCpu, TorchInferenceMachineGenericTests.ComplexNetworkWithSingleParents): pass
class TestNetworkWithMultipleParentsCpu(TestTorchNaiveInferenceMachineBaseCpu, TorchInferenceMachineGenericTests.NetworkWithMultipleParents): pass


# Cuda
class TestNetworkWithSingleParentsCuda(TestTorchNaiveInferenceMachineBaseCuda, TorchInferenceMachineGenericTests.NetworkWithSingleParents): pass
class TestComplexNetworkWithSingleParentsCuda(TestTorchNaiveInferenceMachineBaseCuda, TorchInferenceMachineGenericTests.ComplexNetworkWithSingleParents): pass
class TestNetworkWithMultipleParentsCuda(TestTorchNaiveInferenceMachineBaseCuda, TorchInferenceMachineGenericTests.NetworkWithMultipleParents): pass
