from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Callable, Dict, List, Tuple

import numpy as np

from bayesian_network.bayesian_network import BayesianNetwork
from bayesian_network.inference_machines.abstractions import IInferenceMachine
from bayesian_network.inference_machines.evidence import Evidence, EvidenceLoader
from bayesian_network.optimizers.abstractions import IEvaluator


@dataclass(frozen=True)
class Log:
    ts: datetime
    epoch: int
    iteration: int
    ll: float


class OptimizerLogger:
    def __init__(self, should_log: Callable[[int, int], bool] | None = None):
        self._should_log = should_log
        self._logs: List[Log] = []

    def log(self, epoch: int, iteration: int, ll: float):
        if self._should_log:
            if not self._should_log(epoch, iteration):
                return

        if iteration in self._logs:
            raise RuntimeError(f"A log for iteration {iteration} was already added")

        log = Log(
            datetime.now(),
            epoch,
            iteration,
            ll,
        )

        self._logs.append(log)

        logging.info("%s", log)

    @property
    def logs(self):
        return self._logs


class Evaluator(IEvaluator):
    def __init__(
        self,
        inference_machine_factory: Callable[[BayesianNetwork], IInferenceMachine],
        evidence: Evidence,
        should_evaluate: Callable[[int, int], bool],
    ):
        self._inference_machine_factory = inference_machine_factory
        self._evidence = evidence
        self._should_evaluate = should_evaluate

        self._log_likelihoods: Dict[Tuple[int, int], float] = {}

    def evaluate(self, epoch: int, iteration: int, network: BayesianNetwork):
        if not self._should_evaluate(epoch, iteration):
            return

        inference_machine = self._inference_machine_factory(network)

        inference_machine.enter_evidence(self._evidence)

        ll = inference_machine.log_likelihood()

        self._log_likelihoods[(epoch, iteration)] = ll

        logging.info("Evaluated for epoch %s, iteration: %s - ll: %s", epoch, iteration, ll)

    @property
    def log_likelihoods(self):
        return self._log_likelihoods


class BatchEvaluator(IEvaluator):
    def __init__(
        self,
        inference_machine_factory: Callable[[BayesianNetwork], IInferenceMachine],
        evidence_loader: EvidenceLoader,
        should_evaluate: Callable[[int, int], bool],
    ):
        self._inference_machine_factory = inference_machine_factory
        self._evidence_loader = evidence_loader
        self._should_evaluate = should_evaluate

        self._log_likelihoods: Dict[Tuple[int, int], float] = {}

    def evaluate(self, epoch: int, iteration: int, network: BayesianNetwork):
        if not self._should_evaluate(epoch, iteration):
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
        self._log_likelihoods[(epoch, iteration)] = total_ll

        logging.info("Evaluated for epoch %s, ll: %s", epoch, total_ll)

    @property
    def log_likelihoods(self):
        return self._log_likelihoods
