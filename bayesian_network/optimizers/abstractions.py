from typing import Dict
from bayesian_network.bayesian_network import BayesianNetwork
from bayesian_network.inference_machines.evidence import Evidence, EvidenceBatches


from abc import ABC, abstractmethod


class IOptimizer(ABC):
    @abstractmethod
    def optimize(self, evidence: Evidence) -> None:
        pass


class IBatchOptimizer(ABC):
    @abstractmethod
    def optimize(self, batches: EvidenceBatches) -> None:
        pass


class IEvaluator(ABC):
    @abstractmethod
    def evaluate(self, iteration: int, network: BayesianNetwork):
        pass

    @property
    @abstractmethod
    def log_likelihoods(self) -> Dict[int, float]:
        pass
