from dataclasses import dataclass
from typing import Callable, List, Optional

import torch

from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.inference_machines.abstractions import IInferenceMachine
from bayesian_network.inference_machines.evidence import Evidence
from bayesian_network.inference_machines.spa_v1.factor_graph import FactorGraph
from bayesian_network.inference_machines.common import (
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
            bayesian_network,
            observed_nodes,
            torch_settings=self.settings.torch_settings,
            num_observations=num_observations,
        )
        self.observed_nodes = observed_nodes
        self.num_observations = num_observations
        self.must_iterate: bool = True

    def infer_single_nodes(self, nodes: List[Node]) -> List[torch.Tensor]:
        if self.must_iterate:
            self._iterate()

        return [self._infer_single_node(node) for node in nodes]

    def _infer_single_node(self, node: Node) -> torch.Tensor:
        variable_node = self.factor_graph.variable_nodes[node]
        factor_node = self.factor_graph.factor_nodes[node]

        [value_to_factor_node] = [
            message.value
            for message in variable_node.output_messages
            if message.destination is factor_node
        ]
        [value_from_factor_node] = [
            message.value
            for message in variable_node.input_messages
            if message.source is factor_node
        ]

        p = value_from_factor_node * value_to_factor_node

        return p

    def infer_nodes_with_parents(self, children: List[Node]) -> List[torch.Tensor]:
        if self.must_iterate:
            self._iterate()

        return [self._infer_node_with_parents(child) for child in children]

    def _infer_node_with_parents(self, child: Node) -> torch.Tensor:
        child_factor_node = self.factor_graph.factor_nodes[child]
        input_values = [input_message.value for input_message in child_factor_node.input_messages]

        num_inputs = len(input_values)

        # Example einsum equation for three inputs:
        # a, [..., 0], b, [..., 1], c, [..., 2], d, [..., 0, 1, 2], [..., 0, 1, 2]
        einsum_equation = []
        for index, input_value in enumerate(input_values):
            einsum_equation.append(input_value)
            einsum_equation.append([..., index])

        einsum_equation.append(child_factor_node.cpt)
        einsum_equation.append([..., *range(num_inputs)])
        einsum_equation.append([..., *range(num_inputs)])

        p = torch.einsum(*einsum_equation)

        return p

    def _iterate(self):
        for iteration in range(self.settings.num_iterations):
            self.factor_graph.iterate()

            if self.settings.callback:
                self.settings.callback(iteration)

        self.must_iterate = False

    def enter_evidence(self, evidence: Evidence):
        # evidence.shape: List[num_nodes * [num_observations x num_states]
        self.factor_graph.enter_evidence(evidence.data)

        self.must_iterate = True

    def log_likelihood(self) -> float:
        if not self.observed_nodes:
            raise Exception("Log likelihood can't be calculated with 0 observed nodes")

        if self.must_iterate:
            self._iterate()

        # local_likelihoods: [num_observations, num_nodes]
        local_likelihoods = torch.stack(
            [node.local_likelihood for node in self.factor_graph.variable_nodes.values()], dim=1
        )

        log_likelihoods = torch.log(local_likelihoods)

        if self.settings.average_log_likelihood:
            return log_likelihoods.sum(dim=1).mean().item()
        else:
            return log_likelihoods.sum().item()
