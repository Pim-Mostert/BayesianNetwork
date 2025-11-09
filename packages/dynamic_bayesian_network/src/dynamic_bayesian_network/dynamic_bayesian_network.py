import torch
from bayesian_network.bayesian_network import Node
from common.statistics import is_probability_matrix


class DynamicNode(Node):
    def __init__(self, cpt: torch.Tensor, dcpt: torch.Tensor, name=None):
        if not dcpt | is_probability_matrix():
            raise Exception("The DCPT should sum to 1 along the last dimension.")

        super().__init__(cpt, name)

        self.dcpt = dcpt


class DynamicBayesianNetwork:
    def __init__(
        self,
        nodes: list[DynamicNode],
        parents: dict[DynamicNode, list[DynamicNode]],
        sequential_parents: dict[DynamicNode, list[DynamicNode]],
    ):
        self._nodes = nodes
        self._parents = parents
        self._sequential_parents = sequential_parents
