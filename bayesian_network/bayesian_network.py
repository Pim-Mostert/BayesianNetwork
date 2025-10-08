from abc import ABC, abstractmethod
from typing import Dict, List, Set

import networkx as nx
import torch

from bayesian_network.common.statistics import is_valid_probability_matrix


class Node:
    def __repr__(self):
        return self.name if self.name else super().__repr__()

    def __init__(self, cpt: torch.Tensor, name=None):
        if not is_valid_probability_matrix(cpt):
            raise Exception("The CPT should sum to 1 along the last dimension.")

        self.num_states: int = cpt.shape[-1]
        self.cpt = cpt
        self.name = name


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
        return self._G.nodes

    @property
    def num_nodes(self):
        return self._G.number_of_nodes()

    def parents_of(self, node: Node):
        return self._G.predecessors(node)


class NetworkValidationError(Exception):
    pass


class NetworkValidator(ABC):
    @abstractmethod
    def evaluate(self, nodes: List[Node], parents: Dict[Node, List[Node]]):
        pass


class CPTsMatchParents(NetworkValidator):
    def evaluate(self, nodes: List[Node], parents: Dict[Node, List[Node]]):
        for node in nodes:
            for i, parent in enumerate(parents[node]):
                if node.cpt.shape[i] != parent.num_states:
                    raise NetworkValidationError(
                        f"Node \"{node}\"'s CPT's {i}'th dimension should match with corresponding parent \"{parent}\"'s num_states."
                    )


class NoDuplicateNodes(NetworkValidator):
    def evaluate(self, nodes: List[Node], parents: Dict[Node, List[Node]]):
        nodes_set = set()
        for node in nodes:
            if node in nodes_set:
                raise ValueError(f'Node "{node}" is added more than once.')

            nodes_set.add(node)


class NoIsolatedNodes(NetworkValidator):
    def evaluate(self, nodes: List[Node], parents: Dict[Node, List[Node]]):
        for node in nodes:
            children = [child for child in nodes if node in parents[child]]

            if (not children) and (not parents[node]):
                raise NetworkValidationError(
                    f'Node "{node}" is isolated, i.e. has no children nor parents.'
                )


class ParentsExistInNetwork(NetworkValidator):
    def evaluate(self, nodes: List[Node], parents: Dict[Node, List[Node]]):
        nodes_set = set(nodes)

        for node in nodes_set:
            for parent in parents[node]:
                if parent not in nodes_set:
                    raise NetworkValidationError(
                        f'Parent "{parent}" for node "{node}" is not added.'
                    )


class BayesianNetworkBuilder:
    def __init__(self):
        self.nodes: List[Node] = []
        self.parents: Dict[Node, List[Node]] = {}
        self._validators: Set[NetworkValidator] = {
            CPTsMatchParents(),
            NoDuplicateNodes(),
            NoIsolatedNodes(),
            ParentsExistInNetwork(),
        }

    def add_node(self, node: Node, parents: None | Node | List[Node] = None):
        self.nodes.append(node)
        self.set_parents(node, parents)

        return self

    def set_parents(self, node: Node, parents: None | Node | List[Node] = None):
        if parents:
            if isinstance(parents, Node):
                parents = [parents]

            self.parents[node] = parents
        else:
            self.parents[node] = []

    def build(self):
        # Validate
        for validator in self._validators:
            validator.evaluate(self.nodes, self.parents)

        # Build
        return BayesianNetwork(self.nodes, self.parents)
