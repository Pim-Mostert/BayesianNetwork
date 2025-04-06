from abc import ABC
from typing import List

from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.common.torch_settings import TorchSettings
from bayesian_network.inference_machines.spa_inference_machine_v2 import SpaInferenceMachine
from tests.inference_machines.torch_inference_machine_generic_tests import (
    TorchInferenceMachineGenericTests,
)


# Helper class
class TestSpaInferenceMachineV2Base(ABC):
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
    TestSpaInferenceMachineV2Base,
    TorchInferenceMachineGenericTests.NetworkWithSingleParents,
):
    pass


class TestComplexNetworkWithSingleParents(
    TestSpaInferenceMachineV2Base,
    TorchInferenceMachineGenericTests.ComplexNetworkWithSingleParents,
):
    pass


# class HandleNumericalUnderflow(
#     TestSpaInferenceMachineV2Base,
#     TorchInferenceMachineGenericTests.HandleNumericalUnderflow,
# ):
#     pass
