from typing import List, Callable, Dict

import torch

from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.inference_machines.factor_graph.factor_graph_2 import FactorGraph, FactorNode
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
        self.einsum_equations_per_node: Dict[Node, List] = {
            node: self._construct_einsum_equation_for_node(node)
            for node
            in bayesian_network.nodes
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
        for iteration in range(self.num_iterations):
            self.factor_graph.iterate()

            self.callback(self.factor_graph, iteration)

        self.must_iterate = False

    def enter_evidence(self, evidence: torch.Tensor):
        # evidence.shape: [num_observations x num_observed_nodes], label-encoded
        if evidence.shape[0] != self.num_observations:
            raise Exception(f'First dimension of evidence should match num_observations ({self.num_observations}), but is {evidence.shape[0]}')

        evidence_list: List[torch.Tensor] = []

        for i, observed_node in enumerate(self.observed_nodes):
            e = torch.zeros((self.num_observations, observed_node.num_states), device=self.device, dtype=torch.float64)

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

        log_likelihood = torch.log(self.factor_graph.local_likelihoods).sum()

        return log_likelihood

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


