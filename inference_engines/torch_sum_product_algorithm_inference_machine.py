from typing import List, Callable

import torch

from inference_engines.factor_graph.factor_graph import FactorGraph
from model.bayesian_network import BayesianNetwork
from model.interfaces import IInferenceMachine
from model.nodes import Node


class TorchSumProductAlgorithmInferenceMachine(IInferenceMachine):
    def __init__(self,
                 bayesian_network: BayesianNetwork,
                 observed_nodes: List[Node],
                 device: torch.device,
                 num_iterations: int,
                 num_observations: int,
                 callback: Callable[[FactorGraph, int], None]):
        self.device = device
        self.bayesian_network = bayesian_network
        self.factor_graph = FactorGraph(
            bayesian_network,
            observed_nodes,
            device=device,
            num_observations=num_observations)
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
            message.get_value()
            for message
            in variable_node.output_messages
            if message.destination is factor_node
        ]
        [value_from_factor_node] = [
            message.get_value()
            for message
            in variable_node.input_messages
            if message.source is factor_node
        ]

        p = value_from_factor_node * value_to_factor_node

        # p /= p.sum(dim=1, keepdim=True)

        return p

    def infer_children_with_parents(self, children: List[Node]) -> List[torch.Tensor]:
        if self.must_iterate:
            self._iterate()

        return [self._infer_child_with_parents(child) for child in children]

    def _infer_child_with_parents(self, child: Node) -> torch.Tensor:
        child_factor_node = self.factor_graph.factor_nodes[child]
        input_values = [
            input_message.get_value()
            for input_message
            in child_factor_node.input_messages
        ]

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

        # Normalize
        # sum_dims = tuple(range(1, num_inputs+1))
        # p /= p.sum(axis=sum_dims, keepdims=True)

        return p

    def _iterate(self):
        for iteration in range(self.num_iterations):
            self.factor_graph.iterate()

            self.callback(self.factor_graph, iteration)

        self.must_iterate = False

    def enter_evidence(self, evidence: torch.Tensor):
        # evidence.shape = [num_trials, num_observed_nodes], label-encoded
        evidence_list: List[torch.Tensor] = []

        for i, observed_node in enumerate(self.observed_nodes):
            e = torch.zeros((self.num_observations, observed_node.numK), device=self.device, dtype=torch.float64)

            for n in range(evidence.shape[0]):
                e[n, evidence[n, i]] = 1

            evidence_list.append(e)

        self.factor_graph.enter_evidence(evidence_list)

        self.must_iterate = True

    def log_likelihood(self) -> float:
        if not self.observed_nodes:
            raise Exception("Log likelihood can't be calculated with 0 observed nodes")

        if self.must_iterate:
            self._iterate()

        log_likelihoods = [
            node.log_likelihood
            for node
            in self.factor_graph.variable_nodes.values()
        ]

        log_likelihood_total = torch.stack(log_likelihoods).sum(dim=0)

        return log_likelihood_total