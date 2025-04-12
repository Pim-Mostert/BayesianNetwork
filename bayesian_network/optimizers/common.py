from dataclasses import dataclass
from datetime import datetime
from typing import Dict

import numpy as np

from bayesian_network.optimizers.interfaces import IOptimizerLogger


@dataclass(frozen=True)
class Log:
    ts: datetime
    iteration: int
    ll: float


class OptimizerLogger(IOptimizerLogger):
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
