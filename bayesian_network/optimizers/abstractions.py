from typing import Dict, Tuple

from bayesian_network.bayesian_network import BayesianNetwork
from bayesian_network.inference_machines.evidence import EvidenceLoader, Evidence


from abc import ABC, abstractmethod


class IOptimizer(ABC):
    @abstractmethod
    def optimize(self, evidence: Evidence) -> None:
        raise NotImplementedError()


class IBatchOptimizer(ABC):
    @abstractmethod
    def optimize(self, evidence_loader: EvidenceLoader) -> None:
        raise NotImplementedError()


class IEvaluator(ABC):
    @abstractmethod
    def evaluate(self, epoch: int, iteration: int, network: BayesianNetwork):
        raise NotImplementedError()

    @property
    @abstractmethod
    def log_likelihoods(self) -> Dict[Tuple[int, int], float]:
        raise NotImplementedError()
