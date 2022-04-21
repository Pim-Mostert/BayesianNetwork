from typing import List, Dict

from model.nodes import Node


class BayesianNetwork:
    def __init__(self, nodes: List['Node'], parents: Dict['Node', List['Node']]):
        self.nodes = nodes
        self.parents = parents
        self.num_nodes = len(self.nodes)

        self.children = {
            node: [child for child in self.nodes if node in self.parents[child]]
            for node in self.nodes
        }

        self.root_nodes = [
            node
            for node
            in self.nodes
            if not self.parents[node] and self.children[node]
        ]

        self.leaf_nodes = [
            node
            for node
            in self.nodes
            if self.parents[node] and not self.children[node]
        ]

    def get_children(self, node: Node) -> List[Node]:
        return [child for child in self.parents.keys() if node in self.parents[child]]

    def is_root_node(self, node: Node) -> bool:
        return node in self.root_nodes

    def is_leaf_node(self, node: None) -> bool:
        return node in self.leaf_nodes
