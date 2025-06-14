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
