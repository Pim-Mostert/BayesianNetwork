from typing import List, Dict

import torch
import common as cmn
from model.bayesian_network import BayesianNetwork
from model.nodes import NodeType, Node, CPTNode


class TorchNaiveInferenceMachine:
    def __init__(self, cfg, bayesian_network: BayesianNetwork, observed_nodes: List[Node]):
        self.device = cfg.device

        if not all([node.node_type == NodeType.CPTNode for node in bayesian_network.nodes]):
            raise Exception(f'Only nodes of type {NodeType.CPTNode} supported')

        self.dims = [cptnode.numK for cptnode in bayesian_network.nodes]
        self.num_nodes = len(bayesian_network.nodes)
        self.num_observed_nodes = len(observed_nodes)
        self.node_to_index = {node: bayesian_network.nodes.index(node) for node in bayesian_network.nodes}
        self.observed_nodes_indices: List[CPTNode] = [self.node_to_index[node] for node in observed_nodes]

        self.p = self._calculate_p_complete(bayesian_network.nodes, bayesian_network.parents)[None, ...]

    def _calculate_p_complete(self, cptnodes: List[CPTNode], parents: Dict[CPTNode, List[CPTNode]]):
        dims = [cptnode.numK for cptnode in cptnodes]
        p = torch.ones(dims, device=self.device)

        for cptnode in cptnodes:
            new_shape = [1] * self.num_nodes
            new_shape[self.node_to_index[cptnode]] = cptnode.numK

            for parent in parents[cptnode]:
                parent_index = self.node_to_index[parent]
                new_shape[parent_index] = parent.numK

            p_node = torch.tensor(cptnode.cpt, device=self.device) \
                .reshape(new_shape)

            p *= p_node

        return p

    def enter_evidence(self, evidence: torch.tensor):
        if evidence.shape[1] != self.num_observed_nodes:
            raise Exception(f'Second dimension of evidence must match number of observed nodes: {len(self.observed_nodes_indices)}')

        num_trials = evidence.shape[0]
        dims = [num_trials] + self.dims

        p_evidence = torch.ones(dims, device=self.device)

        for (i, observed_node_index) in enumerate(self.observed_nodes_indices):
            p_evidence_node = self._calculate_p_evidence_for_observed_node(observed_node_index, evidence[:, i])

            node_dims = [1] * self.num_nodes
            node_dims[observed_node_index] = self.dims[observed_node_index]
            node_dims = [num_trials] + node_dims

            p_evidence *= p_evidence_node.reshape(node_dims)

        self.p = self.p * p_evidence

        sum_over_dims = range(1, self.num_nodes+1)
        c = self.p.sum(axis=tuple(sum_over_dims), keepdims=True)
        self.p /= c

        return torch.log(c).mean()

    def _calculate_p_evidence_for_observed_node(self, observed_node_index, evidence):
        num_trials = evidence.shape[0]

        p_evidence = torch.zeros((num_trials, self.dims[observed_node_index]), device=self.device)
        for i_trial in torch.arange(num_trials, device=self.device):
            p_evidence[i_trial, evidence[i_trial]] = 1

        return p_evidence

    def infer(self, nodes):
        node_indices = [self.node_to_index[node] for node in nodes]
        dims = [d+1 for d in range(self.num_nodes) if d not in node_indices]

        if not dims:
            return self.p

        return self.p.sum(axis=dims)
