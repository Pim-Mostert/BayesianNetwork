from dataclasses import dataclass
from datetime import datetime
import logging
import time
from typing import Callable, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from bayesian_network.bayesian_network import BayesianNetwork
from bayesian_network.common.torch_settings import TorchSettings
from bayesian_network.inference_machines.abstractions import IInferenceMachine
from bayesian_network.inference_machines.evidence import Evidence
from bayesian_network.optimizers.abstractions import IEvaluator


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
    def log_likelihoods(self) -> Dict[int, float]:
        return self._log_likelihoods


@dataclass(frozen=True)
class MnistBatchEvaluatorSettings(EvaluatorSettings):
    torch_settings: TorchSettings
    batch_size: int
    gamma: float


class MnistBatchEvaluator(IEvaluator):
    def __init__(
        self,
        settings: MnistBatchEvaluatorSettings,
        inference_machine_factory: Callable[[BayesianNetwork], IInferenceMachine],
    ):
        self._settings = settings
        self._inference_machine_factory = inference_machine_factory
        self._log_likelihoods: Dict[int, float] = {}

        mnist = torchvision.datasets.MNIST(
            "./experiments/mnist",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )

        self._num_samples = len(mnist)

        self._data_loader = DataLoader(
            dataset=mnist,
            batch_size=self._settings.batch_size,
        )

    def _morph_into_evidence(self, data):
        height, width = data.shape[2:4]
        num_features = height * width
        num_observations = data.shape[0]

        # Morph into evidence structure
        data = data.reshape([num_observations, num_features])

        evidence = Evidence(
            [
                torch.stack([1 - x, x]).T
                for x in data.T * (1 - self._settings.gamma) + self._settings.gamma / 2
            ],
            self._settings.torch_settings,
        )

        return evidence

    def evaluate(self, iteration: int, network: BayesianNetwork):
        if not (iteration % self._settings.iteration_interval) == 0:
            return

        inference_machine = self._inference_machine_factory(network)

        start = time.time()
        lls = []
        for _, (batch, _) in enumerate(self._data_loader):
            evidence = self._morph_into_evidence(batch)

            inference_machine.enter_evidence(evidence)

            ll = inference_machine.log_likelihood()

            if inference_machine.settings.average_log_likelihood:
                # TODO:
                # - Deze class generiek maken voor elke DataLoader?
                ll *= len(batch) / self._num_samples

            lls.append(ll)

        total_ll = np.array(lls).sum()
        self._log_likelihoods[iteration] = total_ll

        print(f"It took {time.time() - start} seconds")

        logging.info("Evaluated for iteration %s, ll: %s", iteration, total_ll)

    @property
    def log_likelihoods(self):
        return self._log_likelihoods
