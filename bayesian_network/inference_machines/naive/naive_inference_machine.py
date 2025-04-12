from typing import Dict, List

import torch

from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.inference_machines.evidence import Evidence
from bayesian_network.inference_machines.common import IInferenceMachine, InferenceMachineSettings


class NaiveInferenceMachine(IInferenceMachine):
    def __init__(
        self,
        settings: InferenceMachineSettings,
        bayesian_network: BayesianNetwork,
        observed_nodes: List[Node],
    ):
        self.settings = settings

        self.dims = [node.num_states for node in bayesian_network.nodes]
        self.num_nodes = len(bayesian_network.nodes)
        self.num_observed_nodes = len(observed_nodes)
        self.node_to_index = {
            node: bayesian_network.nodes.index(node) for node in bayesian_network.nodes
        }
        self.observed_nodes_indices: List[int] = [
            self.node_to_index[node] for node in observed_nodes
        ]
        self.bayesian_network = bayesian_network
        self._log_likelihoods = None

        self.p = self._calculate_p_complete(bayesian_network.nodes, bayesian_network.parents)[
            None, ...
        ]

    def _calculate_p_complete(self, nodes: List[Node], parents: Dict[Node, List[Node]]):
        dims = [node.num_states for node in nodes]
        p = torch.ones(
            dims,
            dtype=self.settings.torch_settings.dtype,
            device=self.settings.torch_settings.device,
        )

        for node in nodes:
            new_shape = [1] * self.num_nodes
            new_shape[self.node_to_index[node]] = node.num_states

            for parent in parents[node]:
                parent_index = self.node_to_index[parent]
                new_shape[parent_index] = parent.num_states

            p_node = node.cpt.reshape(new_shape)

            p *= p_node

        return p

    def enter_evidence(self, evidence: Evidence):
        if evidence.num_observed_nodes != self.num_observed_nodes:
            raise Exception(
                "Length of evidence must match number of observed"
                " nodes: {len(self.observed_nodes_indices)}"
            )

        num_trials = evidence.num_observations
        dims = [num_trials] + self.dims

        p_evidence = torch.ones(
            dims,
            dtype=self.settings.torch_settings.dtype,
            device=self.settings.torch_settings.device,
        )

        for i, observed_node_index in enumerate(self.observed_nodes_indices):
            node_dims = [1] * self.num_nodes
            node_dims[observed_node_index] = self.dims[observed_node_index]
            node_dims = [num_trials] + node_dims

            p_evidence *= evidence.data[i].reshape(node_dims)

        self.p = self.p * p_evidence

        sum_over_dims = range(1, self.num_nodes + 1)
        c = self.p.sum(dim=tuple(sum_over_dims), keepdim=True)
        self.p /= c

        self._log_likelihoods = torch.log(c)

    def _infer(self, nodes):
        node_indices = [self.node_to_index[node] for node in nodes]
        dims = [d + 1 for d in range(self.num_nodes) if d not in node_indices]

        if not dims:
            return self.p

        return self.p.sum(dim=dims)

    def infer_nodes_with_parents(self, child_nodes: List[Node]):
        p = [self._infer(self.bayesian_network.parents[node] + [node]) for node in child_nodes]

        return p

    def infer_single_nodes(self, nodes: List[Node]):
        p = [self._infer([node]) for node in nodes]

        return p

    def log_likelihood(self) -> torch.Tensor:
        if self.num_observed_nodes == 0:
            raise Exception("Log likelihood can't be calculated with 0 observed nodes")

        if self.settings.average_log_likelihood:
            return self._log_likelihoods.mean()
        else:
            return self._log_likelihoods.sum()
