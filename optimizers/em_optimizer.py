from typing import Callable, Dict

import torch

from model.bayesian_network import BayesianNetwork
from model.interfaces import IOptimizer, IInferenceMachine
from model.nodes import Node


class EmOptimizer(IOptimizer):
    def __init__(self, bayesian_network: BayesianNetwork, inference_machine_factory: Callable[[BayesianNetwork], IInferenceMachine]):
        self.bayesian_network = bayesian_network
        self.inference_machine_factory = inference_machine_factory

    def optimize(self, evidence, num_iterations, iteration_callback):
        for iteration in range(num_iterations):
            inference_machine = self.inference_machine_factory(self.bayesian_network)
            inference_machine.enter_evidence(evidence)
            ll = inference_machine.log_likelihood()

            p_conditionals = self._e_step(inference_machine)

            self._m_step(p_conditionals)

            iteration_callback(ll, iteration)

    def _e_step(self, inference_machine: IInferenceMachine) -> Dict[Node, torch.Tensor]:
        p_conditionals = {
            node: p_conditional
            for node, p_conditional in
            zip(self.bayesian_network.nodes, inference_machine.infer_nodes_with_parents(self.bayesian_network.nodes))
        }

        return p_conditionals

    def _m_step(self, p_conditionals: Dict[Node, torch.Tensor]):
        for node in self.bayesian_network.nodes:
            cpt = p_conditionals[node].mean(dim=0)

            cpt /= cpt.sum(dim=-1, keepdims=True)

            node.cpt = cpt
