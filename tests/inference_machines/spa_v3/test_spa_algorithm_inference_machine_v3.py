from abc import ABC
from typing import List

from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.common.torch_settings import TorchSettings
from bayesian_network.inference_machines.spa_v3.spa_inference_machine import SpaInferenceMachine  # noqa
from tests.inference_machines.torch_inference_machine_generic_tests import (
    TorchInferenceMachineGenericTests,
)


# Helper class
class TestSpaInferenceMachineV3Base(ABC):
    def get_torch_settings(self) -> TorchSettings:
        return TorchSettings()

    def create_inference_machine(
        self,
        bayesian_network: BayesianNetwork,
        observed_nodes: List[Node],
        num_observations: int,
    ):
        return SpaInferenceMachine(
            bayesian_network=bayesian_network,
            observed_nodes=observed_nodes,
            torch_settings=self.get_torch_settings(),
            num_iterations=20,
            num_observations=num_observations,
            callback=lambda factor_graph, iteration: None,
        )


# Actual tests
class TestNetworkWithSingleParents(
    TestSpaInferenceMachineV3Base,
    TorchInferenceMachineGenericTests.NetworkWithSingleParents,
):
    pass


class TestComplexNetworkWithSingleParents(
    TestSpaInferenceMachineV3Base,
    TorchInferenceMachineGenericTests.ComplexNetworkWithSingleParents,
):
    pass


class HandleNumericalUnderflow(
    TestSpaInferenceMachineV3Base,
    TorchInferenceMachineGenericTests.HandleNumericalUnderflow,
):
    pass
