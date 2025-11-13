from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping

from dynamic_bayesian_network.dynamic_bayesian_network import DynamicBayesianNetwork, Node


class NetworkValidationError(Exception):
    pass


class NetworkValidator(ABC):
    @abstractmethod
    def evaluate(
        self,
        nodes: Iterable[Node],
        parents: Mapping[Node, Iterable[Node]],
        sequential_parents: Mapping[Node, Iterable[Node]],
    ):
        pass


class CPTsMatchParents(NetworkValidator):
    def evaluate(
        self,
        nodes: Iterable[Node],
        parents: Mapping[Node, Iterable[Node]],
        sequential_parents: Mapping[Node, Iterable[Node]],
    ):
        for node in nodes:
            node_parents = (*sequential_parents[node], *parents[node])

            for i, parent in enumerate(node_parents):
                if node.cpt.shape[i] != parent.num_states:
                    raise NetworkValidationError(
                        f"Node \"{node}\"'s CPT's {i}'th dimension should match with corresponding (sequential) parent \"{parent}\"'s num_states."
                    )


class NoDuplicateNodes(NetworkValidator):
    def evaluate(
        self,
        nodes: Iterable[Node],
        parents: Mapping[Node, Iterable[Node]],
        sequential_parents: Mapping[Node, Iterable[Node]],
    ):
        nodes_set = set()
        for node in nodes:
            if node in nodes_set:
                raise ValueError(f'Node "{node}" is added more than once.')

            nodes_set.add(node)


class NoIsolatedNodes(NetworkValidator):
    def evaluate(
        self,
        nodes: Iterable[Node],
        parents: Mapping[Node, Iterable[Node]],
        sequential_parents: Mapping[Node, Iterable[Node]],
    ):
        for node in nodes:
            children = [child for child in nodes if node in parents[child]]

            if (not children) and (not parents[node]):
                raise NetworkValidationError(
                    f'Node "{node}" is isolated, i.e. has no children nor parents.'
                )


class ParentsExistInNetwork(NetworkValidator):
    def evaluate(
        self,
        nodes: Iterable[Node],
        parents: Mapping[Node, Iterable[Node]],
        sequential_parents: Mapping[Node, Iterable[Node]],
    ):
        nodes_set = set(nodes)

        for node in nodes_set:
            for parent in parents[node]:
                if parent not in nodes_set:
                    raise NetworkValidationError(
                        f'Parent "{parent}" for node "{node}" is not added.'
                    )


class SequentialParentsExistInNetwork(NetworkValidator):
    def evaluate(
        self,
        nodes: Iterable[Node],
        parents: Mapping[Node, Iterable[Node]],
        sequential_parents: Mapping[Node, Iterable[Node]],
    ):
        nodes_set = set(nodes)

        for node in nodes_set:
            for sequential_parent in sequential_parents[node]:
                if sequential_parent not in nodes_set:
                    raise NetworkValidationError(
                        f'Sequential parent "{sequential_parent}" for node "{node}" is not added.'
                    )


class NoDuplicateNodeNames(NetworkValidator):
    def evaluate(
        self,
        nodes: Iterable[Node],
        parents: Mapping[Node, Iterable[Node]],
        sequential_parents: Mapping[Node, Iterable[Node]],
    ):
        names = set()
        for node in nodes:
            if not node.name:
                continue

            if node.name in names:
                raise NetworkValidationError(f'A node with name "{node.name}" was already added.')

            names.add(node.name)


class NoDuplicateParents(NetworkValidator):
    def evaluate(
        self,
        nodes: Iterable[Node],
        parents: Mapping[Node, Iterable[Node]],
        sequential_parents: Mapping[Node, Iterable[Node]],
    ):
        for node in nodes:
            parents_set = set()

            for parent in parents[node]:
                if parent in parents_set:
                    raise NetworkValidationError(
                        f"Node {node}'s parent {parent}'s is added as parent more than once."
                    )

                parents_set.add(parent)


class NoDuplicateSequentialParents(NetworkValidator):
    def evaluate(
        self,
        nodes: Iterable[Node],
        parents: Mapping[Node, Iterable[Node]],
        sequential_parents: Mapping[Node, Iterable[Node]],
    ):
        for node in nodes:
            sequential_parents_set = set()

            for sequential_parent in sequential_parents[node]:
                if sequential_parent in sequential_parents_set:
                    raise NetworkValidationError(
                        f"Node {node}'s parent {sequential_parent}'s is added as parent more than once."
                    )

                sequential_parents_set.add(sequential_parent)


class PriorShouldMatchCPT(NetworkValidator):
    def evaluate(
        self,
        nodes: Iterable[Node],
        parents: Mapping[Node, Iterable[Node]],
        sequential_parents: Mapping[Node, Iterable[Node]],
    ):
        for node in nodes:
            num_parents = len(list(parents[node]))

            if not node.prior.size() == node.cpt.size()[-num_parents:]:
                raise NetworkValidationError(f"Node {node}'s prior does not match its CPT")


class DynamicBayesianNetworkBuilder:
    def __init__(self):
        self._nodes: list[Node] = []
        self._parents: dict[Node, list[Node]] = {}
        self._sequential_parents: dict[Node, list[Node]] = {}
        self._network_validators: set[NetworkValidator] = {
            CPTsMatchParents(),
            NoDuplicateNodes(),
            NoIsolatedNodes(),
            ParentsExistInNetwork(),
            SequentialParentsExistInNetwork(),
            NoDuplicateNodeNames(),
            NoDuplicateParents(),
            NoDuplicateSequentialParents(),
        }

    def add_node(
        self,
        node: Node,
        parents: None | Node | list[Node] = None,
        sequential_parents: None | Node | list[Node] = None,
    ):
        self._nodes.append(node)

        if parents:
            self.set_parents(node, parents)

        if sequential_parents:
            self.set_sequential_parents(node, sequential_parents)

        return self

    def add_nodes(
        self,
        nodes: list[Node],
        parents: None | Node | list[Node] = None,
        sequential_parents: None | Node | list[Node] = None,
    ):
        for node in nodes:
            self.add_node(node, parents, sequential_parents)

        return self

    def set_parents(
        self,
        node: Node,
        parents: Node | list[Node],
    ):
        if isinstance(parents, Node):
            parents = [parents]

        self._parents[node] = parents

        return self

    def set_sequential_parents(
        self,
        node: Node,
        sequential_parents: Node | list[Node],
    ):
        if isinstance(sequential_parents, Node):
            sequential_parents = [sequential_parents]

        self._sequential_parents[node] = sequential_parents

        return self

    def build(self):
        # Validate
        for validator in self._network_validators:
            validator.evaluate(self._nodes, self._parents, self._sequential_parents)

        # Build
        return DynamicBayesianNetwork(self._nodes, self._parents, self._sequential_parents)
