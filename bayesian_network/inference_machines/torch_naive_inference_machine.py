from typing import List, Dict

import torch

from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.interfaces import IInferenceMachine


class TorchNaiveInferenceMachine(IInferenceMachine):
    def __init__(self, bayesian_network: BayesianNetwork, observed_nodes: List[Node], device: torch.device):
        self.device = device

        self.dims = [node.num_states for node in bayesian_network.nodes]
        self.num_nodes = len(bayesian_network.nodes)
        self.num_observed_nodes = len(observed_nodes)
        self.node_to_index = {node: bayesian_network.nodes.index(node) for node in bayesian_network.nodes}
        self.observed_nodes_indices: List[Node] = [self.node_to_index[node] for node in observed_nodes]
        self.bayesian_network = bayesian_network
        self._log_likelihood = None

        self.p = self._calculate_p_complete(bayesian_network.nodes, bayesian_network.parents)[None, ...]

    def _calculate_p_complete(self, nodes: List[Node], parents: Dict[Node, List[Node]]):
        dims = [node.num_states for node in nodes]
        p = torch.ones(dims, dtype=torch.float64, device=self.device)

        for node in nodes:
            new_shape = [1] * self.num_nodes
            new_shape[self.node_to_index[node]] = node.num_states

            for parent in parents[node]:
                parent_index = self.node_to_index[parent]
                new_shape[parent_index] = parent.num_states

            p_node = node.cpt.reshape(new_shape)

            p *= p_node

        return p

    def enter_evidence(self, evidence: List[torch.tensor]):
        if len(evidence) != self.num_observed_nodes:
            raise Exception(f'Length of evidence must match number of observed nodes: {len(self.observed_nodes_indices)}')

        num_trials = evidence[0].shape[0]
        dims = [num_trials] + self.dims

        p_evidence = torch.ones(dims, device=self.device)

        for (i, observed_node_index) in enumerate(self.observed_nodes_indices):
            node_dims = [1] * self.num_nodes
            node_dims[observed_node_index] = self.dims[observed_node_index]
            node_dims = [num_trials] + node_dims

            p_evidence *= evidence[i].reshape(node_dims)

        self.p = self.p * p_evidence

        sum_over_dims = range(1, self.num_nodes+1)
        c = self.p.sum(axis=tuple(sum_over_dims), keepdims=True)
        self.p /= c

        self._log_likelihood = torch.log(c).sum()

    def _infer(self, nodes):
        node_indices = [self.node_to_index[node] for node in nodes]
        dims = [d+1 for d in range(self.num_nodes) if d not in node_indices]

        if not dims:
            return self.p

        return self.p.sum(dim=dims)

    def infer_nodes_with_parents(self, child_nodes: List[Node]):
        p = [
            self._infer(self.bayesian_network.parents[node] + [node])
            for node
            in child_nodes
        ]

        return p

    def infer_single_nodes(self, nodes: List[Node]):
        p = [
            self._infer([node])
            for node
            in nodes
        ]

        return p

    def log_likelihood(self) -> float:
        if self.num_observed_nodes == 0:
            raise Exception("Log likelihood can't be calculated with 0 observed nodes")

        return self._log_likelihood
