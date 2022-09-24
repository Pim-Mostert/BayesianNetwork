from typing import Callable, List

import torch
import time

from bayesian_network.bayesian_network import BayesianNetwork
from bayesian_network.interfaces import IOptimizer, IInferenceMachine


class EmOptimizer(IOptimizer):
    def __init__(self, 
                 bayesian_network: BayesianNetwork, 
                 inference_machine_factory: Callable[[BayesianNetwork], IInferenceMachine]):
        self.bayesian_network = bayesian_network
        self.inference_machine_factory = inference_machine_factory

    def optimize(self, evidence, num_iterations, iteration_callback):
        for iteration in range(num_iterations):
            start_time = time.time()

            # Construct inference machine and enter evidence
            inference_machine = self.inference_machine_factory(self.bayesian_network)
            inference_machine.enter_evidence(evidence)
            ll = inference_machine.log_likelihood()

            # E-step
            p_conditionals = self._e_step(inference_machine)

            # M-step
            self._m_step(p_conditionals)

            # User feedback
            duration = time.time() - start_time
            iteration_callback(ll, iteration, duration)

    def _e_step(self, inference_machine: IInferenceMachine) -> List[torch.Tensor]:
        # List[torch.Tensor((observations x parent1 x parent2 x ... x child))]
        p_all = inference_machine.infer_nodes_with_parents(self.bayesian_network.nodes)

        # Average over observations
        p_conditionals = [
            p.mean(dim=0)
            for p
            in p_all
        ]

        return p_conditionals

    def _m_step(self, p_conditionals: List[torch.Tensor]):
        for node, p_conditional in zip(self.bayesian_network.nodes, p_conditionals):
            # Normalize to conditional probability distribution
            cpt = p_conditional / p_conditional.sum(dim=-1, keepdim=True)

            # Update node
            node.cpt = cpt
