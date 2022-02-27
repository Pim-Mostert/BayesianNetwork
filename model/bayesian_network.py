from typing import List, Dict

from model.nodes import Node


class BayesianNetwork:
    def __init__(self, nodes: List['Node'], parents: Dict['Node', List['Node']]):
        self.nodes = nodes
        self.parents = parents
        self.num_nodes = len(self.nodes)


