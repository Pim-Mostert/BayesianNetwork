from abc import ABC
from typing import List

from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.common.torch_settings import TorchSettings
from bayesian_network.inference_machines.interfaces import IInferenceMachine
from bayesian_network.inference_machines.naive_inference_machine import NaiveInferenceMachine
from tests.inference_machines.torch_inference_machine_generic_tests import (
    TorchInferenceMachineGenericTests,
)


# Helper class
class TestTorchNaiveInferenceMachineBase(ABC):
    def get_torch_settings(self) -> TorchSettings:
        return TorchSettings()

    def create_inference_machine(
        self,
        bayesian_network: BayesianNetwork,
        observed_nodes: List[Node],
        num_observations: int,
    ) -> IInferenceMachine:
        return NaiveInferenceMachine(
            bayesian_network=bayesian_network,
            observed_nodes=observed_nodes,
            torch_settings=self.get_torch_settings(),
        )


# Actual tests
class TestNetworkWithSingleParents(
    TestTorchNaiveInferenceMachineBase,
    TorchInferenceMachineGenericTests.NetworkWithSingleParents,
):
    pass


class TestComplexNetworkWithSingleParents(
    TestTorchNaiveInferenceMachineBase,
    TorchInferenceMachineGenericTests.ComplexNetworkWithSingleParents,
):
    pass


class TestNetworkWithMultipleParents(
    TestTorchNaiveInferenceMachineBase,
    TorchInferenceMachineGenericTests.NetworkWithMultipleParents,
):
    pass
