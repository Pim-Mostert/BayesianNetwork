from abc import ABC, abstractmethod
from typing import List

import torch

from bayesian_network.bayesian_network import Node
from bayesian_network.inference_machines.evidence import Evidence


class IInferenceMachine(ABC):
    @abstractmethod
    def enter_evidence(self, evidence: Evidence) -> None:
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
