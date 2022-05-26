from abc import abstractmethod, ABC
from typing import List

import torch

from common.utilities import Cfg
from inference_engines.tests.torch_inference_machine_base_tests import TorchInferenceMachineBaseTests
from inference_engines.torch_naive_inference_machine import TorchNaiveInferenceMachine
from model.bayesian_network import BayesianNetwork
from model.nodes import Node


# Helper classes
class TorchNaiveInferenceMachineTestBase:
    class TorchNaiveInferenceMachineTestBase(ABC):
        @abstractmethod
        def get_device(self) -> str:
            pass

        def create_inference_machine(self,
                                     bayesian_network: BayesianNetwork,
                                     observed_nodes: List[Node],
                                     num_observations: int):
            return TorchNaiveInferenceMachine(
                cfg=Cfg({'device': self.get_device()}),
                bayesian_network=bayesian_network,
                observed_nodes=observed_nodes)

    class TorchNaiveInferenceMachineTestBaseCpu(TorchNaiveInferenceMachineTestBase):
        def get_device(self) -> str:
            return 'cpu'

    class TorchNaiveInferenceMachineTestBaseCuda(TorchNaiveInferenceMachineTestBase):
        def setUp(self):
            if not torch.cuda.is_available():
                self.skipTest('Cuda not available')

        def get_device(self) -> str:
            return 'cuda'


# Run all tests for cpu
class TestNetworkWithSingleParentsCpu(TorchNaiveInferenceMachineTestBase.TorchNaiveInferenceMachineTestBaseCpu,
                                      TorchInferenceMachineBaseTests.NetworkWithSingleParents):
    pass


class TestNetworkWithMultipleParentsCpu(TorchNaiveInferenceMachineTestBase.TorchNaiveInferenceMachineTestBaseCpu,
                                        TorchInferenceMachineBaseTests.NetworkWithMultipleParents):
    pass


# Run all tests for cuda
class TestNetworkWithSingleParentsCuda(TorchNaiveInferenceMachineTestBase.TorchNaiveInferenceMachineTestBaseCuda,
                                       TorchInferenceMachineBaseTests.NetworkWithSingleParents):
    pass


class TestNetworkWithMultipleParentsCuda(TorchNaiveInferenceMachineTestBase.TorchNaiveInferenceMachineTestBaseCuda,
                                         TorchInferenceMachineBaseTests.NetworkWithMultipleParents):
    pass
