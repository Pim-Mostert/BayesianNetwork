from abc import ABC, abstractmethod
from typing import List

import torch

from bayesian_network.bayesian_network import Node


class IBayesianNetworkSampler(ABC):
    @abstractmethod
    def sample(self, num_samples: int, nodes: List[Node]):
        pass


class IInferenceMachine(ABC):
    @abstractmethod
    def enter_evidence(self, evidence) -> None:
        pass

    @abstractmethod
    def infer_single_nodes(self, nodes: List[Node]) -> List[torch.Tensor]:
        pass

    @abstractmethod
    def infer_nodes_with_parents(self, child_nodes: List[Node]) -> List[torch.Tensor]:
        pass

    @abstractmethod
    def log_likelihood(self) -> float:
        pass


class IOptimizer(ABC):
    @abstractmethod
    def optimize(self, evidence, num_iterations: int, iteration_callback) -> None:
        pass

