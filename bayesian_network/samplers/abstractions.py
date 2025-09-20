from abc import ABC, abstractmethod
from typing import List

from bayesian_network.bayesian_network import Node


class IBayesianNetworkSampler(ABC):
    @abstractmethod
    def sample(self, num_samples: int, nodes: List[Node]):
        raise NotImplementedError()
