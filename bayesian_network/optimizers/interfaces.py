from abc import ABC, abstractmethod

from bayesian_network.inference_machines.evidence import Evidence, EvidenceBatches


class IBatchOptimizer(ABC):
    @abstractmethod
    def optimize(self, batches: EvidenceBatches) -> None:
        pass


class IOptimizer(ABC):
    @abstractmethod
    def optimize(self, evidence: Evidence) -> None:
        pass
