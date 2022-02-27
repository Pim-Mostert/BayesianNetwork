from abc import ABC, abstractmethod
from typing import List

from model.nodes import Node


class ISampler(ABC):
    @abstractmethod
    def sample(self, num_samples: int, nodes: List[Node]):
        pass


class IInferenceMachine(ABC):
    @abstractmethod
    def enter_evidence(self, evidence):
        pass

    @abstractmethod
    def infer(self, nodes: List[Node]):
        pass


class IOptimizer(ABC):
    @abstractmethod
    def optimize(self, evidence, num_iterations: int, iteration_callback):
        pass

