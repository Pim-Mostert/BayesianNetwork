from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch

from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.inference_machines.evidence import Evidence
from bayesian_network.inference_machines.spa_v2.factor_graph import FactorGraph
from bayesian_network.inference_machines.common import (
    IInferenceMachine,
    InferenceMachineSettings,
)


@dataclass
class SpaInferenceMachineSettings(InferenceMachineSettings):
    num_iterations: int
    callback: Optional[Callable[[int], None]] = None


class SpaInferenceMachine(IInferenceMachine):
    def __init__(
        self,
        settings: SpaInferenceMachineSettings,
        bayesian_network: BayesianNetwork,
        observed_nodes: List[Node],
        num_observations: int,
    ):
        self.settings = settings
        self.factor_graph = FactorGraph(
            bayesian_network=bayesian_network,
            observed_nodes=observed_nodes,
            torch_settings=self.settings.torch_settings,
            num_observations=num_observations,
        )
        self.observed_nodes = observed_nodes
        self.num_observations = num_observations
        self.must_iterate: bool = True
        self.einsum_equations_per_node: Dict[Node, List] = {
            node: self._construct_einsum_equation_for_node(node) for node in bayesian_network.nodes
        }

    def infer_single_nodes(self, nodes: List[Node]) -> List[torch.Tensor]:
        if self.must_iterate:
            self._iterate()

        return [self._infer_single_node(node) for node in nodes]

    def _infer_single_node(self, node: Node) -> torch.Tensor:
        variable_node = self.factor_graph.variable_nodes[node]

        value_to_factor_node = variable_node.output_with_indices_to_local_factor_node.output
        value_from_factor_node = variable_node.input_from_local_factor_node

        p = value_from_factor_node * value_to_factor_node

        return p

    def infer_nodes_with_parents(self, nodes: List[Node]) -> List[torch.Tensor]:
        if self.must_iterate:
            self._iterate()

        return [self._infer_node_with_parents(node) for node in nodes]

    def _infer_node_with_parents(self, node: Node) -> torch.Tensor:
        p = torch.einsum(*self.einsum_equations_per_node[node])

        return p

    def _iterate(self):
        for iteration in range(self.settings.num_iterations):
            self.factor_graph.iterate()

            if self.settings.callback:
                self.settings.callback(iteration)

        self.must_iterate = False

    def enter_evidence(self, evidence: Evidence):
        self.factor_graph.enter_evidence(evidence.data)

        self.must_iterate = True

    def log_likelihood(self) -> torch.Tensor:
        if not self.observed_nodes:
            raise Exception("Log likelihood can't be calculated with 0 observed nodes")

        if self.must_iterate:
            self._iterate()

        log_likelihoods = torch.log(self.factor_graph.local_likelihoods)

        if self.settings.average_log_likelihood:
            return log_likelihoods.sum(dim=1).mean()
        else:
            return log_likelihoods.sum()

    def _construct_einsum_equation_for_node(self, node: Node) -> List:
        factor_node = self.factor_graph.factor_nodes[node]
        num_inputs = len(factor_node.all_inputs)

        # Example einsum equation for three inputs:
        # 'ni, nj, nk, ijk->nijk', x1, x2, x3, cpt
        # x1, [..., 0], x2, [..., 1], x3, [..., 2], cpt, [..., 0, 1, 2], [..., 0, 1, 2]
        einsum_equation = []

        # Each input used to calculate current output
        for index, input in enumerate(factor_node.all_inputs):
            einsum_equation.append(input)
            einsum_equation.append([..., index])

        # Cpt of the factor node
        einsum_equation.append(factor_node.cpt)
        einsum_equation.append([..., *range(num_inputs)])

        # Desired output dimensions
        einsum_equation.append([..., *range(num_inputs)])

        return einsum_equation
