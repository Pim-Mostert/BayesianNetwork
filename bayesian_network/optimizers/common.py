from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Callable, Dict

import numpy as np

from bayesian_network.bayesian_network import BayesianNetwork
from bayesian_network.inference_machines.abstractions import IInferenceMachine
from bayesian_network.inference_machines.evidence import Evidence, EvidenceLoader
from bayesian_network.optimizers.abstractions import IEvaluator


@dataclass(frozen=True)
class Log:
    ts: datetime
    iteration: int
    ll_train: float | None
    ll_eval: float | None


class OptimizerLogger:
    def __init__(self):
        self._logs: Dict[int, Log] = {}

    def log_iteration(
        self,
        iteration: int,
        ll_train: float | None,
        ll_eval: float | None,
    ):
        if iteration in self._logs:
            raise RuntimeError(f"A log for iteration {iteration} was already added")

        log = Log(
            datetime.now(),
            iteration,
            ll_train,
            ll_eval,
        )

        self._logs[iteration] = log

        logging.info("%s", log)

    @property
    def log_likelihoods(self):
        iterations = sorted(self._logs)
        log_likelihoods = [self._logs[iteration].ll for iteration in iterations]

        return np.array(iterations), np.array(log_likelihoods)


@dataclass(frozen=True)
class EvaluatorSettings:
    iteration_interval: int


class Evaluator(IEvaluator):
    def __init__(
        self,
        settings: EvaluatorSettings,
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

    @property
    def log_likelihoods(self):
        iterations = sorted(self._log_likelihoods)
        log_likelihoods = [self._log_likelihoods[iteration] for iteration in iterations]

        return np.array(iterations), np.array(log_likelihoods)


class BatchEvaluator(IEvaluator):
    def __init__(
        self,
        settings: EvaluatorSettings,
        inference_machine_factory: Callable[[BayesianNetwork], IInferenceMachine],
        evidence_loader: EvidenceLoader,
    ):
        self._settings = settings
        self._inference_machine_factory = inference_machine_factory
        self._evidence_loader = evidence_loader

        self._log_likelihoods: Dict[int, float] = {}

    def evaluate(self, iteration: int, network: BayesianNetwork):
        if not (iteration % self._settings.iteration_interval) == 0:
            return

        inference_machine = self._inference_machine_factory(network)

        lls = []
        for evidence in iter(self._evidence_loader):
            inference_machine.enter_evidence(evidence)

            ll = inference_machine.log_likelihood()

            if inference_machine.settings.average_log_likelihood:
                ll *= len(evidence) / self._evidence_loader.num_observations

            lls.append(ll)

        total_ll = np.array(lls).sum()
        self._log_likelihoods[iteration] = total_ll

        logging.info("Evaluated for iteration %s, ll: %s", iteration, total_ll)

    @property
    def log_likelihoods(self):
        iterations = sorted(self._log_likelihoods)
        log_likelihoods = [self._log_likelihoods[iteration] for iteration in iterations]

        return np.array(iterations), np.array(log_likelihoods)
