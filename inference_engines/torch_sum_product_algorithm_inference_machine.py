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
                 callback: Callable[[FactorGraph], None]):
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

    def infer(self, nodes: List[Node]) -> torch.Tensor:
        for _ in range(self.num_iterations):
            self.factor_graph.iterate()

            self.callback(self.factor_graph)

        if len(nodes) > 2:
            raise Exception("Only inference on single nodes or two neighbouring nodes supported")

        if len(nodes) == 1:
            return self._infer_single_node(nodes[0])
        else:
            return self._infer_neighbouring_nodes(nodes)

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

        p /= p.sum(dims=1, keepDims=True)

        return p

    def _infer_neighbouring_nodes(self, nodes: List[Node]) -> torch.Tensor:
        if len(nodes) == 2 and not self.bayesian_network.are_neighbours(nodes[0], nodes[1]):
            raise Exception("Only inference on single nodes or two neighbouring nodes supported")

        raise Exception("todo")

    def enter_evidence(self, evidence: torch.Tensor):
        # evidence.shape = [num_trials, num_observed_nodes], label-encoded
        evidence_list: List[torch.Tensor] = []

        for i, observed_node in enumerate(self.observed_nodes):
            e = torch.zeros((self.num_observations, observed_node.numK), device=self.device, dtype=torch.float64)

            for n in range(evidence.shape[0]):
                e[n, evidence[n, i]] = 1

            evidence_list.append(e)

        self.factor_graph.enter_evidence(evidence_list)
