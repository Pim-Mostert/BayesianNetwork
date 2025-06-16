from typing import Dict, List

import torch

from bayesian_network.common.statistics import is_valid_probability_matrix


class Node:
    def __repr__(self):
        return super().__repr__() if self.name is None else f"{type(self).__name__} - {self.name}"

    def __init__(self, cpt: torch.Tensor, name=None):
        if not is_valid_probability_matrix(cpt):
            raise Exception("The CPT should sum to 1 along the last dimension")

        self.num_states: int = cpt.shape[-1]
        self.cpt = cpt
        self.name = name


class BayesianNetwork:
    def __init__(self, nodes: List[Node], parents: Dict[Node, List[Node]]):
        self.nodes = nodes
        self.parents = parents
        self.num_nodes = len(self.nodes)

        self.children: Dict[Node, List[Node]] = {
            node: [child for child in self.nodes if node in self.parents[child]]
            for node in self.nodes
        }

        self.leaf_nodes = [
            node for node in self.nodes if self.parents[node] and not self.children[node]
        ]

    def is_leaf_node(self, node: Node) -> bool:
        return node in self.leaf_nodes
