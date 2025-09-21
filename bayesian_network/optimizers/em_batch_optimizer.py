from dataclasses import dataclass
from typing import Callable, List, Optional

import torch

from bayesian_network.bayesian_network import BayesianNetwork
from bayesian_network.inference_machines.abstractions import IInferenceMachine
from bayesian_network.inference_machines.evidence import EvidenceLoader
from bayesian_network.optimizers.abstractions import IBatchOptimizer, IEvaluator
from bayesian_network.optimizers.common import (
    OptimizerLogger,
)


@dataclass(frozen=True)
class EmBatchOptimizerSettings:
    learning_rate: float
    num_epochs: int


class EmBatchOptimizer(IBatchOptimizer):
    def __init__(
        self,
        bayesian_network: BayesianNetwork,
        inference_machine_factory: Callable[[BayesianNetwork], IInferenceMachine],
        settings: EmBatchOptimizerSettings,
        logger: Optional[OptimizerLogger] = None,
        evaluator: Optional[IEvaluator] = None,
    ):
        self._bayesian_network = bayesian_network
        self._inference_machine_factory = inference_machine_factory
        self._settings = settings
        self._logger = logger
        self._evaluator = evaluator

    def optimize(self, evidence_loader: EvidenceLoader):
        for epoch in range(self._settings.num_epochs):
            for iteration, evidence in enumerate(evidence_loader):
                # Construct inference machine
                inference_machine = self._inference_machine_factory(self._bayesian_network)

                # Enter evidence
                inference_machine.enter_evidence(evidence)
                ll = inference_machine.log_likelihood()

                # E-step
                p_conditionals = self._e_step(inference_machine)

                # M-step
                self._m_step(p_conditionals)

                # Log iteration
                if self._logger:
                    self._logger.log(epoch, iteration, ll)

                # Evaluate
                if self._evaluator:
                    self._evaluator.evaluate(epoch, iteration, self._bayesian_network)

    def _e_step(self, inference_machine: IInferenceMachine) -> List[torch.Tensor]:
        # List[torch.Tensor((observations x parent1 x parent2 x ... x child))]
        p_all = inference_machine.infer_nodes_with_parents(self._bayesian_network.nodes)

        # Average over observations
        p_conditionals = [p.mean(dim=0) for p in p_all]

        return p_conditionals

    def _m_step(self, p_conditionals: List[torch.Tensor]):
        for node, p_conditional in zip(self._bayesian_network.nodes, p_conditionals):
            # Normalize to conditional probability distribution
            cpt = p_conditional / p_conditional.sum(dim=-1, keepdim=True)

            # Update node according to learning rate
            lr = self._settings.learning_rate
            node.cpt = (1 - lr) * node.cpt + lr * cpt
