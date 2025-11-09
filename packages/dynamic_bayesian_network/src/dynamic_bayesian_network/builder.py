from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import cast

from bayesian_network.bayesian_network import Node
from bayesian_network.builder import (
    CPTsMatchParents,
    NetworkValidator,
    NoDuplicateNodeNames,
    NoDuplicateNodes,
    NoIsolatedNodes,
    ParentsExistInNetwork,
)

from packages.dynamic_bayesian_network.src.dynamic_bayesian_network.dynamic_bayesian_network import (
    DynamicBayesianNetwork,
    DynamicNode,
)


class NetworkValidationError(Exception):
    pass


class DynamicNetworkValidator(ABC):
    @abstractmethod
    def evaluate(
        self,
        nodes: Iterable[DynamicNode],
        parents: Mapping[DynamicNode, Iterable[DynamicNode]],
        sequential_parents: Mapping[DynamicNode, Iterable[DynamicNode]],
    ):
        pass


class DCPTsMatchParents(DynamicNetworkValidator):
    def evaluate(
        self,
        nodes: Iterable[DynamicNode],
        parents: Mapping[DynamicNode, Iterable[DynamicNode]],
        sequential_parents: Mapping[DynamicNode, Iterable[DynamicNode]],
    ):
        for node in nodes:
            all_parents = (*sequential_parents[node], *parents[node])

            for i, parent in enumerate(all_parents):
                if node.dcpt.shape[i] != parent.num_states:
                    raise NetworkValidationError(
                        f"Node \"{node}\"'s DCPT's {i}'th dimension should match with corresponding (sequential) parent \"{parent}\"'s num_states."
                    )


class SequentialParentsExistInNetwork(DynamicNetworkValidator):
    def evaluate(
        self,
        nodes: Iterable[DynamicNode],
        parents: Mapping[DynamicNode, Iterable[DynamicNode]],
        sequential_parents: Mapping[DynamicNode, Iterable[DynamicNode]],
    ):
        nodes_set = set(nodes)

        for node in nodes_set:
            for sequential_parent in sequential_parents[node]:
                if sequential_parent not in nodes_set:
                    raise NetworkValidationError(
                        f'Sequential parent "{sequential_parent}" for node "{node}" is not added.'
                    )


class DynamicBayesianNetworkBuilder:
    def __init__(self):
        self.nodes: list[DynamicNode] = []
        self.parents: dict[DynamicNode, list[DynamicNode]] = {}
        self.sequential_parents: dict[DynamicNode, list[DynamicNode]] = {}
        self._network_validators: set[NetworkValidator] = {
            CPTsMatchParents(),
            NoDuplicateNodes(),
            NoIsolatedNodes(),
            ParentsExistInNetwork(),
            NoDuplicateNodeNames(),
        }
        self._dynamic_network_validators: set[DynamicNetworkValidator] = {
            DCPTsMatchParents(),
            SequentialParentsExistInNetwork(),
        }

    def add_node(self, node: DynamicNode, parents: None | DynamicNode | list[DynamicNode] = None):
        self.nodes.append(node)
        self.set_parents(node, parents)

        return self

    def add_nodes(
        self, nodes: list[DynamicNode], parents: None | DynamicNode | list[DynamicNode] = None
    ):
        for node in nodes:
            self.nodes.append(node)
            self.set_parents(node, parents)

        return self

    def set_parents(
        self, node: DynamicNode, parents: None | DynamicNode | list[DynamicNode] = None
    ):
        if parents:
            if isinstance(parents, Node):
                parents = [parents]

            self.parents[node] = parents
        else:
            self.parents[node] = []

    def build(self):
        # Validate
        for validator in self._network_validators:
            validator.evaluate(self.nodes, cast(Mapping[Node, Iterable[Node]], self.parents))

        for validator in self._dynamic_network_validators:
            validator.evaluate(self.nodes, self.parents, self.sequential_parents)

        # Build
        return DynamicBayesianNetwork(self.nodes, self.parents, self.sequential_parents)
