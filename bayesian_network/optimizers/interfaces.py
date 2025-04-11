from abc import ABC, abstractmethod

import numpy as np

from bayesian_network.inference_machines.evidence import Evidence, EvidenceBatches


class IBatchOptimizer(ABC):
    @abstractmethod
    def optimize(self, batches: EvidenceBatches) -> None:
        pass


class IOptimizer(ABC):
    @abstractmethod
    def optimize(self, evidence: Evidence) -> None:
        pass


class IOptimizerLogger(ABC):
    @abstractmethod
    def log_iteration(
        self,
        iteration: int,
        ll: float,
    ):
        pass

    @abstractmethod
    def get_loglikelihood(self) -> np.ndarray:
        pass
