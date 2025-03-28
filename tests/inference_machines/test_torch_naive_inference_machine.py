from abc import ABC, abstractmethod
from typing import List

import torch

from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.common.torch_settings import TorchSettings
from bayesian_network.inference_machines.torch_naive_inference_machine import TorchNaiveInferenceMachine
from bayesian_network.interfaces import IInferenceMachine
from tests.inference_machines.torch_inference_machine_generic_tests import TorchInferenceMachineGenericTests


# Helper classes
class TestTorchNaiveInferenceMachineBase(ABC):
    @abstractmethod
    def get_torch_settings(self) -> TorchSettings:
        pass

    def create_inference_machine(
        self,
        bayesian_network: BayesianNetwork,
        observed_nodes: List[Node],
        num_observations: int,
    ) -> IInferenceMachine:
        return TorchNaiveInferenceMachine(
            bayesian_network=bayesian_network,
            observed_nodes=observed_nodes,
            torch_settings=self.get_torch_settings(),
        )


class TestTorchNaiveInferenceMachineBaseCpu(TestTorchNaiveInferenceMachineBase, ABC):
    def get_torch_settings(self) -> TorchSettings:
        return TorchSettings(torch.device("cpu"), torch.double)


class TestTorchNaiveInferenceMachineBaseCuda(TestTorchNaiveInferenceMachineBase, ABC):
    def get_torch_settings(self) -> TorchSettings:
        return TorchSettings(torch.device("cuda"), torch.double)


class TestTorchNaiveInferenceMachineBaseMps(TestTorchNaiveInferenceMachineBase, ABC):
    def get_torch_settings(self) -> TorchSettings:
        return TorchSettings(torch.device("mps"), torch.float32)


# Actual tests
# Cpu
class TestNetworkWithSingleParentsCpu(
    TestTorchNaiveInferenceMachineBaseCpu,
    TorchInferenceMachineGenericTests.NetworkWithSingleParents,
):
    pass


class TestComplexNetworkWithSingleParentsCpu(
    TestTorchNaiveInferenceMachineBaseCpu,
    TorchInferenceMachineGenericTests.ComplexNetworkWithSingleParents,
):
    pass


class TestNetworkWithMultipleParentsCpu(
    TestTorchNaiveInferenceMachineBaseCpu,
    TorchInferenceMachineGenericTests.NetworkWithMultipleParents,
):
    pass


# Cuda
class TestNetworkWithSingleParentsCuda(
    TestTorchNaiveInferenceMachineBaseCuda,
    TorchInferenceMachineGenericTests.NetworkWithSingleParents,
):
    pass


class TestComplexNetworkWithSingleParentsCuda(
    TestTorchNaiveInferenceMachineBaseCuda,
    TorchInferenceMachineGenericTests.ComplexNetworkWithSingleParents,
):
    pass


class TestNetworkWithMultipleParentsCuda(
    TestTorchNaiveInferenceMachineBaseCuda,
    TorchInferenceMachineGenericTests.NetworkWithMultipleParents,
):
    pass


# Mps
class TestNetworkWithSingleParentsMps(
    TestTorchNaiveInferenceMachineBaseMps,
    TorchInferenceMachineGenericTests.NetworkWithSingleParents,
):
    pass


class TestComplexNetworkWithSingleParentsMps(
    TestTorchNaiveInferenceMachineBaseMps,
    TorchInferenceMachineGenericTests.ComplexNetworkWithSingleParents,
):
    pass


class TestNetworkWithMultipleParentsMps(
    TestTorchNaiveInferenceMachineBaseMps,
    TorchInferenceMachineGenericTests.NetworkWithMultipleParents,
):
    pass
