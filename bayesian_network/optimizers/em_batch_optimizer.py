from dataclasses import dataclass
from typing import Callable, List, Optional

import torch

from bayesian_network.bayesian_network import BayesianNetwork
from bayesian_network.inference_machines.evidence import Evidence
from bayesian_network.interfaces import IInferenceMachine, IOptimizer


@dataclass(frozen=True)
class EmBatchOptimizerSettings:
    num_iterations: int
    batch_size: int
    learning_rate: float
    iteration_callback: Optional[Callable[[int, float, BayesianNetwork], None]] = None


class EmBatchOptimizer(IOptimizer):
    def __init__(
        self,
        bayesian_network: BayesianNetwork,
        inference_machine_factory: Callable[[BayesianNetwork], IInferenceMachine],
        settings: EmBatchOptimizerSettings,
    ):
        self.bayesian_network = bayesian_network
        self.inference_machine_factory = inference_machine_factory
        self.settings = settings

    def optimize(self, evidence: Evidence):
        for iteration in range(self.settings.num_iterations):
            # Construct inference machine
            inference_machine = self.inference_machine_factory(self.bayesian_network)

            # Select batch
            # evidence: List[(num_observed_nodes)], torch.Tensor: [num_observations, num_states]

            #
            inference_machine.enter_evidence(evidence)
            ll = inference_machine.log_likelihood()

            # E-step
            p_conditionals = self._e_step(inference_machine)

            # M-step
            self._m_step(p_conditionals)

            # User feedback
            if self.settings.iteration_callback:
                self.settings.iteration_callback(iteration, ll, self.bayesian_network)

    def _e_step(self, inference_machine: IInferenceMachine) -> List[torch.Tensor]:
        # List[torch.Tensor((observations x parent1 x parent2 x ... x child))]
        p_all = inference_machine.infer_nodes_with_parents(self.bayesian_network.nodes)

        # Average over observations
        p_conditionals = [p.mean(dim=0) for p in p_all]

        return p_conditionals

    def _m_step(self, p_conditionals: List[torch.Tensor]):
        for node, p_conditional in zip(self.bayesian_network.nodes, p_conditionals):
            # Normalize to conditional probability distribution
            cpt = p_conditional / p_conditional.sum(dim=-1, keepdim=True)

            # Update node
            node.cpt = cpt
