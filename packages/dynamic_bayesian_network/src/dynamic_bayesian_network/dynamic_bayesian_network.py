from bayesian_network.bayesian_network import Node


class DynamicBayesianNetwork:
    def __init__(
        self,
        nodes: list[Node],
        parents: dict[Node, list[Node]],
        sequential_parents: dict[Node, list[Node]],
    ):
        self._nodes = nodes
        self._parents = parents
        self._sequential_parents = sequential_parents
