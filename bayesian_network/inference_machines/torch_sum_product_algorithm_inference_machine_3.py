import itertools
from typing import List, Callable, Tuple
from torch.nn.functional import one_hot

import torch

from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.inference_machines.factor_graph.factor_graph_3 import FactorGraph
from bayesian_network.interfaces import IInferenceMachine


class TorchSumProductAlgorithmInferenceMachine(IInferenceMachine):
    def __init__(self,
                 bayesian_network: BayesianNetwork,
                 observed_nodes: List[Node],
                 device: torch.device,
                 num_iterations: int,
                 num_observations: int,
                 callback: Callable[[FactorGraph, int], None]):
        self.device = device
        self.factor_graph = FactorGraph(
            bayesian_network=bayesian_network,
            observed_nodes=observed_nodes,
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
        variable_node_group = self.factor_graph.get_variable_node_group(node)
        factor_node_group = self.factor_graph.get_factor_node_group(node)

        variable_node_tensor = variable_node_group.get_input_tensor(node, node)
        factor_node_tensor = factor_node_group.get_input_tensor(node, node)

        p = variable_node_tensor * factor_node_tensor

        return p

    def infer_nodes_with_parents(self, nodes: List[Node]) -> List[torch.Tensor]:
        if self.must_iterate:
            self._iterate()

        return [self._infer_node_with_parents(node) for node in nodes]

    def _infer_node_with_parents(self, node: Node) -> torch.Tensor:
        factor_node_group = self.factor_graph.get_factor_node_group(node)

        einsum_equation = []
        for index, input in enumerate(factor_node_group.get_node_inputs(node)):
            einsum_equation.append(input)
            einsum_equation.append([0, index+1])

        einsum_equation.append(factor_node_group.node_cpts[node])
        einsum_equation.append(range(1, factor_node_group._num_inputs+1))

        einsum_equation.append(range(0, factor_node_group._num_inputs+1))

        p = torch.einsum(*einsum_equation)

        return p

    def _iterate(self):
        for iteration in range(self.num_iterations):
            self.factor_graph.iterate()

            self.callback(self.factor_graph, iteration)

        self.must_iterate = False

    def enter_evidence(self, evidence: torch.Tensor):
        # evidence.shape: [num_observations x num_observed_nodes], label-encoded
        if evidence.shape[0] != self.num_observations:
            raise Exception(f'First dimension of evidence should match num_observations ({self.num_observations}), but is {evidence.shape[0]}')
        
        evidence_list = [
            one_hot(evidence, node.num_states).double()
            for evidence, node
            in zip(evidence.long().transpose(0, 1), self.observed_nodes)
        ]

        # evidence: List[(num_nodes)], torch.Tensor: [num_observations, num_states]
        self.factor_graph.enter_evidence(evidence_list)

        self.must_iterate = True

    def log_likelihood(self) -> float:
        if not self.observed_nodes:
            raise Exception("Log likelihood can't be calculated with 0 observed nodes")

        if self.must_iterate:
            self._iterate()

        local_log_likelihoods = [
            variable_node_group.local_log_likelihoods
            for variable_node_group
            in self.factor_graph.variable_node_groups
        ]
        log_likelihood = torch.cat(local_log_likelihoods).sum()

        return log_likelihood


