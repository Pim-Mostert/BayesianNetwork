from abc import ABC, abstractmethod

from bayesian_network.bayesian_network import Node


class NetworkValidationError(Exception):
    pass


class DynamicNetworkValidator(ABC):
    @abstractmethod
    def evaluate(
        self,
        nodes: list[Node],
        parents: dict[Node, list[Node]],
        sequential_parents: dict[Node, list[Node]],
    ):
        pass


# class DynamicBayesianNetworkBuilder:
#     def __init__(self):
#         self.nodes: List[Node] = []
#         self.parents: Dict[Node, List[Node]] = {}
#         self._validators: Set[NetworkValidator] = {
#             CPTsMatchParents(),
#             NoDuplicateNodes(),
#             NoIsolatedNodes(),
#             ParentsExistInNetwork(),
#             NoDuplicateNodeNames(),
#         }

#     def add_node(self, node: Node, parents: None | Node | List[Node] = None):
#         self.nodes.append(node)
#         self.set_parents(node, parents)

#         return self

#     def add_nodes(self, nodes: List[Node], parents: None | Node | List[Node] = None):
#         for node in nodes:
#             self.nodes.append(node)
#             self.set_parents(node, parents)

#         return self

#     def set_parents(self, node: Node, parents: None | Node | List[Node] = None):
#         if parents:
#             if isinstance(parents, Node):
#                 parents = [parents]

#             self.parents[node] = parents
#         else:
#             self.parents[node] = []

#     def build(self):
#         # Validate
#         for validator in self._validators:
#             validator.evaluate(self.nodes, self.parents)

#         # Build
#         return BayesianNetwork(self.nodes, self.parents)
