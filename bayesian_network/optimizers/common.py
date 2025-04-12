from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict

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


@dataclass(frozen=True)
class Log:
    ts: datetime
    iteration: int
    ll: float


class OptimizerLogger:
    def __init__(self):
        self._logs: Dict[int, Log] = {}

    def log_iteration(self, iteration: int, ll: float):
        if iteration in self._logs:
            raise RuntimeError(f"A log for iteration {iteration} was already added")

        log = Log(
            datetime.now(),
            iteration,
            ll,
        )

        print(log)
        self._logs[iteration] = log

    def get_loglikelihood(self):
        return np.array([self._logs[iteration].ll for iteration in sorted(self._logs)])
