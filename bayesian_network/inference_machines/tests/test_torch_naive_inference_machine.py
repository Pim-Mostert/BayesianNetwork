from abc import abstractmethod, ABC
from typing import List

import torch

from bayesian_network.inference_machines.tests.torch_inference_machine_base_tests import TorchInferenceMachineBaseTests
from bayesian_network.inference_machines.torch_naive_inference_machine import TorchNaiveInferenceMachine
from bayesian_network.bayesian_network import BayesianNetwork, Node


# Helper classes
class TorchNaiveInferenceMachineTestBase:
    class TorchNaiveInferenceMachineTestBase(ABC):
        @abstractmethod
        def get_device(self) -> torch.device:
            pass

        def create_inference_machine(self,
                                     bayesian_network: BayesianNetwork,
                                     observed_nodes: List[Node],
                                     num_observations: int):
            return TorchNaiveInferenceMachine(
                bayesian_network=bayesian_network,
                observed_nodes=observed_nodes,
                device=self.get_device())

    class TorchNaiveInferenceMachineTestBaseCpu(TorchNaiveInferenceMachineTestBase):
        def get_device(self) -> torch.device:
            return torch.device('cpu')

    class TorchNaiveInferenceMachineTestBaseCuda(TorchNaiveInferenceMachineTestBase):
        def setUp(self):
            if not torch.cuda.is_available():
                self.skipTest('Cuda not available')

        def get_device(self) -> torch.device:
            return torch.device('cuda')


# Run all tests for cpu
class TestNetworkWithSingleParentsCpu(TorchNaiveInferenceMachineTestBase.TorchNaiveInferenceMachineTestBaseCpu,
                                      TorchInferenceMachineBaseTests.NetworkWithSingleParents):
    pass


class TestComplexNetworkWithSingleParentsCpu(TorchNaiveInferenceMachineTestBase.TorchNaiveInferenceMachineTestBaseCpu,
                                             TorchInferenceMachineBaseTests.ComplexNetworkWithSingleParents):
    pass


class TestNetworkWithMultipleParentsCpu(TorchNaiveInferenceMachineTestBase.TorchNaiveInferenceMachineTestBaseCpu,
                                        TorchInferenceMachineBaseTests.NetworkWithMultipleParents):
    pass


# Run all tests for cuda
class TestNetworkWithSingleParentsCuda(TorchNaiveInferenceMachineTestBase.TorchNaiveInferenceMachineTestBaseCuda,
                                       TorchInferenceMachineBaseTests.NetworkWithSingleParents):
    pass


class TestComplexNetworkWithSingleParentsCuda(TorchNaiveInferenceMachineTestBase.TorchNaiveInferenceMachineTestBaseCuda,
                                              TorchInferenceMachineBaseTests.ComplexNetworkWithSingleParents):
    pass


class TestNetworkWithMultipleParentsCuda(TorchNaiveInferenceMachineTestBase.TorchNaiveInferenceMachineTestBaseCuda,
                                         TorchInferenceMachineBaseTests.NetworkWithMultipleParents):
    pass
