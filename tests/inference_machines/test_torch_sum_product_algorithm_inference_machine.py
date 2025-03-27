from abc import ABC, abstractmethod
from typing import List

import torch

from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.common.torch_settings import TorchSettings
from bayesian_network.inference_machines.torch_sum_product_algorithm_inference_machine import (
    TorchSumProductAlgorithmInferenceMachine,
)
from tests.inference_machines.torch_inference_machine_generic_tests import (
    TorchInferenceMachineGenericTests,
)


# Helper classes
class TestTorchSumProductAlgorithmInferenceMachineBase(ABC):
    @abstractmethod
    def get_torch_settings(self) -> TorchSettings:
        pass

    def create_inference_machine(
        self,
        bayesian_network: BayesianNetwork,
        observed_nodes: List[Node],
        num_observations: int,
    ):
        return TorchSumProductAlgorithmInferenceMachine(
            bayesian_network=bayesian_network,
            observed_nodes=observed_nodes,
            torch_settings=self.get_torch_settings(),
            num_iterations=20,
            num_observations=num_observations,
            callback=lambda factor_graph, iteration: None,
        )


class TestTorchSumProductAlgorithmInferenceMachineBaseCpu(
    TestTorchSumProductAlgorithmInferenceMachineBase, ABC
):
    def get_torch_settings(self) -> TorchSettings:
        return TorchSettings(torch.device("cpu"), torch.double)


class TestTorchSumProductAlgorithmInferenceMachineBaseCuda(
    TestTorchSumProductAlgorithmInferenceMachineBase, ABC
):
    def get_torch_settings(self) -> TorchSettings:
        return TorchSettings(torch.device("cuda"), torch.double)


class TestTorchSumProductAlgorithmInferenceMachineBaseMps(
    TestTorchSumProductAlgorithmInferenceMachineBase, ABC
):
    def get_torch_settings(self) -> TorchSettings:
        return TorchSettings(torch.device("mps"), torch.float32)


# Actual tests
# Cpu
class TestNetworkWithSingleParentsCpu(
    TestTorchSumProductAlgorithmInferenceMachineBaseCpu,
    TorchInferenceMachineGenericTests.NetworkWithSingleParents,
):
    pass


class TestComplexNetworkWithSingleParentsCpu(
    TestTorchSumProductAlgorithmInferenceMachineBaseCpu,
    TorchInferenceMachineGenericTests.ComplexNetworkWithSingleParents,
):
    pass


class HandleNumericalUnderflowCpu(
    TestTorchSumProductAlgorithmInferenceMachineBaseCpu,
    TorchInferenceMachineGenericTests.HandleNumericalUnderflow,
):
    pass


# Cuda
class TestNetworkWithSingleParentsCuda(
    TestTorchSumProductAlgorithmInferenceMachineBaseCuda,
    TorchInferenceMachineGenericTests.NetworkWithSingleParents,
):
    pass


class TestComplexNetworkWithSingleParentsCuda(
    TestTorchSumProductAlgorithmInferenceMachineBaseCuda,
    TorchInferenceMachineGenericTests.ComplexNetworkWithSingleParents,
):
    pass


class HandleNumericalUnderflowCuda(
    TestTorchSumProductAlgorithmInferenceMachineBaseCuda,
    TorchInferenceMachineGenericTests.HandleNumericalUnderflow,
):
    pass


# Mps
class TestNetworkWithSingleParentsMps(
    TestTorchSumProductAlgorithmInferenceMachineBaseMps,
    TorchInferenceMachineGenericTests.NetworkWithSingleParents,
):
    pass


class TestComplexNetworkWithSingleParentsMps(
    TestTorchSumProductAlgorithmInferenceMachineBaseMps,
    TorchInferenceMachineGenericTests.ComplexNetworkWithSingleParents,
):
    pass
