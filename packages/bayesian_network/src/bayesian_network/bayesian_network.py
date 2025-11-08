from typing import Dict, List

import networkx as nx
import torch
from common.statistics import generate_random_probability_matrix, is_valid_probability_matrix
from common.torch_settings import TorchSettings


class Node:
    def __repr__(self):
        return self.name if self.name else super().__repr__()

    def __init__(self, cpt: torch.Tensor, name=None):
        if not is_valid_probability_matrix(cpt):
            raise Exception("The CPT should sum to 1 along the last dimension.")

        self.num_states: int = cpt.shape[-1]
        self.cpt = cpt
        self.name = name

    @classmethod
    def random(cls, size, torch_settings: TorchSettings, name: str | None = None):
        cpt = generate_random_probability_matrix(size, torch_settings)

        return Node(cpt, name)


class BayesianNetwork:
    def __init__(self, nodes: List[Node], parents: Dict[Node, List[Node]]):
        self._G: nx.DiGraph[Node] = nx.DiGraph()

        for node in nodes:
            self._G.add_node(node)

            for parent in parents[node]:
                self._G.add_edge(parent, node)

    @property
    def G(self):
        return self._G

    @property
    def nodes(self):
        return iter(self._G.nodes)

    @property
    def edges(self):
        return iter(self._G.edges)

    @property
    def num_nodes(self):
        return self._G.number_of_nodes()

    @property
    def degrees_of_freedom(self):
        def df_for_node(node: Node):
            s = torch.tensor(node.cpt.shape)
            s[-1] -= 1

            return s.prod().item()

        return sum([df_for_node(node) for node in self.nodes])

    def parents_of(self, node: Node):
        return self._G.predecessors(node)

    def is_leaf_node(self, node: Node):
        return self._G.out_degree(node) == 0

    def children_of(self, node: Node):
        return self._G.successors(node)
