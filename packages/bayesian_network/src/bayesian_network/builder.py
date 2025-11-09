from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping

from bayesian_network.bayesian_network import BayesianNetwork, Node


class NetworkValidationError(Exception):
    pass


class NetworkValidator(ABC):
    @abstractmethod
    def evaluate(self, nodes: Iterable[Node], parents: Mapping[Node, Iterable[Node]]):
        pass


class CPTsMatchParents(NetworkValidator):
    def evaluate(self, nodes: Iterable[Node], parents: Mapping[Node, Iterable[Node]]):
        for node in nodes:
            for i, parent in enumerate(parents[node]):
                if node.cpt.shape[i] != parent.num_states:
                    raise NetworkValidationError(
                        f"Node \"{node}\"'s CPT's {i}'th dimension should match with corresponding parent \"{parent}\"'s num_states."
                    )


class NoDuplicateNodes(NetworkValidator):
    def evaluate(self, nodes: Iterable[Node], parents: Mapping[Node, Iterable[Node]]):
        nodes_set = set()
        for node in nodes:
            if node in nodes_set:
                raise ValueError(f'Node "{node}" is added more than once.')

            nodes_set.add(node)


class NoIsolatedNodes(NetworkValidator):
    def evaluate(self, nodes: Iterable[Node], parents: Mapping[Node, Iterable[Node]]):
        for node in nodes:
            children = [child for child in nodes if node in parents[child]]

            if (not children) and (not parents[node]):
                raise NetworkValidationError(
                    f'Node "{node}" is isolated, i.e. has no children nor parents.'
                )


class ParentsExistInNetwork(NetworkValidator):
    def evaluate(self, nodes: Iterable[Node], parents: Mapping[Node, Iterable[Node]]):
        nodes_set = set(nodes)

        for node in nodes_set:
            for parent in parents[node]:
                if parent not in nodes_set:
                    raise NetworkValidationError(
                        f'Parent "{parent}" for node "{node}" is not added.'
                    )


class NoDuplicateNodeNames(NetworkValidator):
    def evaluate(self, nodes: Iterable[Node], parents: Mapping[Node, Iterable[Node]]):
        names = set()
        for node in nodes:
            if node.name in names:
                raise ValueError(f'A node with name "{node.name}" was already added.')

            names.add(node.name)


class BayesianNetworkBuilder:
    def __init__(self):
        self.nodes: list[Node] = []
        self.parents: dict[Node, list[Node]] = {}
        self._validators: set[NetworkValidator] = {
            CPTsMatchParents(),
            NoDuplicateNodes(),
            NoIsolatedNodes(),
            ParentsExistInNetwork(),
            NoDuplicateNodeNames(),
        }

    def add_node(self, node: Node, parents: None | Node | list[Node] = None):
        self.nodes.append(node)
        self.set_parents(node, parents)

        return self

    def add_nodes(self, nodes: list[Node], parents: None | Node | list[Node] = None):
        for node in nodes:
            self.nodes.append(node)
            self.set_parents(node, parents)

        return self

    def set_parents(self, node: Node, parents: None | Node | list[Node] = None):
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
