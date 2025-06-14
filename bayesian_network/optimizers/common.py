from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Callable, Dict

import numpy as np

from bayesian_network.bayesian_network import BayesianNetwork
from bayesian_network.inference_machines.abstractions import IInferenceMachine
from bayesian_network.inference_machines.evidence import Evidence


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

        self._logs[iteration] = log

        logging.info("%s", log)

    def get_log_likelihood(self):
        return np.array([self._logs[iteration].ll for iteration in sorted(self._logs)])


@dataclass(frozen=True)
class OptimizationEvaluatorSettings:
    iteration_interval: int


class OptimizationEvaluator:
    def __init__(
        self,
        settings: OptimizationEvaluatorSettings,
        inference_machine_factory: Callable[[BayesianNetwork], IInferenceMachine],
        evidence: Evidence,
    ):
        self._settings = settings
        self._inference_machine_factory = inference_machine_factory
        self._evidence = evidence

        self._log_likelihoods: Dict[int, float] = {}

    def evaluate(self, iteration: int, network: BayesianNetwork):
        if not (iteration % self._settings.iteration_interval) == 0:
            return

        inference_machine = self._inference_machine_factory(network)

        inference_machine.enter_evidence(self._evidence)

        ll = inference_machine.log_likelihood()

        self._log_likelihoods[iteration] = ll

        logging.info("Evaluated for iteration %s, ll: %s", iteration, ll)

    def get_log_likelihood(self) -> np.ndarray:
        return np.array(
            [self._log_likelihoods[iteration] for iteration in sorted(self._log_likelihoods)]
        )
