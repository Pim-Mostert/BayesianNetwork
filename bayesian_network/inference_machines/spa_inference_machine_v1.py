from typing import Callable, List

import torch

from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.common.torch_settings import TorchSettings
from bayesian_network.inference_machines.evidence import Evidence
from bayesian_network.inference_machines.factor_graph.factor_graph_v1 import FactorGraph
from bayesian_network.interfaces import IInferenceMachine


class SpaInferenceMachine(IInferenceMachine):
    def __init__(
        self,
        bayesian_network: BayesianNetwork,
        observed_nodes: List[Node],
        torch_settings: TorchSettings,
        num_iterations: int,
        num_observations: int,
        callback: Callable[[FactorGraph, int], None],
    ):
        self.torch_settings = torch_settings
        self.factor_graph = FactorGraph(
            bayesian_network,
            observed_nodes,
            torch_settings=torch_settings,
            num_observations=num_observations,
        )
        self.num_iterations = num_iterations
        self.callback = callback
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
        for iteration in range(self.num_iterations):
            self.factor_graph.iterate()

            self.callback(self.factor_graph, iteration)

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

        log_likelihood_total = torch.log(local_likelihoods).sum()

        return log_likelihood_total
