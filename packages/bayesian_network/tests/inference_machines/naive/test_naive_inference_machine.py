from abc import ABC
from typing import List

from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.common.torch_settings import TorchSettings
from bayesian_network.inference_machines.abstractions import IInferenceMachine
from bayesian_network.inference_machines.common import InferenceMachineSettings
from bayesian_network.inference_machines.naive.naive_inference_machine import NaiveInferenceMachine
from tests.inference_machines.torch_inference_machine_generic_tests import (
    TorchInferenceMachineGenericTests,
)


# Helper class
class TestTorchNaiveInferenceMachineBase(ABC):
    def get_torch_settings(self) -> TorchSettings:
        return TorchSettings(
            device="cpu",
            dtype="float64",
        )

    def create_inference_machine(
        self,
        settings: InferenceMachineSettings,
        bayesian_network: BayesianNetwork,
        observed_nodes: List[Node],
        num_observations: int,
    ) -> IInferenceMachine:
        return NaiveInferenceMachine(
            settings=settings,
            bayesian_network=bayesian_network,
            observed_nodes=observed_nodes,
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
