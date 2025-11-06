from abc import ABC
from dataclasses import asdict
from typing import List

from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.common.torch_settings import TorchSettings
from bayesian_network.inference_machines.common import InferenceMachineSettings
from bayesian_network.inference_machines.spa_v1_1.spa_inference_machine import (
    SpaInferenceMachine,
    SpaInferenceMachineSettings,
)
from tests.inference_machines.torch_inference_machine_generic_tests import (
    TorchInferenceMachineGenericTests,
)


# Helper class
class TestSpaInferenceMachineV1Base(ABC):
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
    ):
        return SpaInferenceMachine(
            settings=SpaInferenceMachineSettings(
                **asdict(settings),
                num_iterations=20,
            ),
            bayesian_network=bayesian_network,
            observed_nodes=observed_nodes,
            num_observations=num_observations,
        )


# Actual tests
class TestNetworkWithSingleParents(
    TestSpaInferenceMachineV1Base,
    TorchInferenceMachineGenericTests.NetworkWithSingleParents,
):
    pass


class TestComplexNetworkWithSingleParents(
    TestSpaInferenceMachineV1Base,
    TorchInferenceMachineGenericTests.ComplexNetworkWithSingleParents,
):
    pass


class HandleNumericalUnderflow(
    TestSpaInferenceMachineV1Base,
    TorchInferenceMachineGenericTests.HandleNumericalUnderflow,
):
    pass
