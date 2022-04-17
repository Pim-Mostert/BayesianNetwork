from typing import List, Dict

import torch

from common.utilities import Cfg
from model.bayesian_network import BayesianNetwork
from model.interfaces import IInferenceMachine
from model.nodes import Node, NodeType


class TorchSumProductAlgorithmInferenceMachine(IInferenceMachine):
    def __init__(self, cfg: Cfg, bayesian_network: BayesianNetwork, observed_nodes: List[Node]):
        self.factor_graph = FactorGraph(bayesian_network)

    def infer(self, nodes: List[Node]):
        pass

    def enter_evidence(self, evidence):
        pass


class FactorGraph:
    def __init__(self, bayesian_network: BayesianNetwork):
        # Instantiate nodes
        self.variable_nodes = {node: VariableNode(node, name=node.name) for node in bayesian_network.nodes}
        self.factor_nodes = {node: FactorNode(node, name=node.name) for node in bayesian_network.nodes}

        # Output messages
        for node in bayesian_network.nodes:
            variable_node = self.variable_nodes[node]
            factor_node = self.factor_nodes[node]

            # Variable nodes
            message = Message()
            variable_node.add_output_message(factor_node, message)

            for child in bayesian_network.children[node]:
                message = Message()
                variable_node.add_output_message(self.factor_nodes[child], message)

            # Factor nodes
            message = Message()
            factor_node.add_output_message(variable_node, message)

            for parent in bayesian_network.parents[node]:
                message = Message()
                factor_node.add_output_message(self.variable_nodes[parent], message)

        # Connect input messages
        for node in bayesian_network.nodes:
            variable_node = self.variable_nodes[node]
            factor_node = self.factor_nodes[node]

            # Variable nodes
            input_message = factor_node.output_messages[variable_node]
            variable_node.add_input_message(input_message, factor_node)

            for child in bayesian_network.children[node]:
                input_message = self.factor_nodes[child].output_messages[variable_node]
                variable_node.add_input_message(input_message, child)

            # Factor nodes
            input_message = variable_node.output_messages[factor_node]
            factor_node.add_input_message(input_message, variable_node)

            for parent in bayesian_network.parents[node]:
                input_message = self.variable_nodes[parent].output_messages[factor_node]
                factor_node.add_input_message(input_message, parent)


class Message:
    def __init__(self):
        pass


class NodeBase:
    def __repr__(self):
        if self.name is None:
            return super().__repr__()
        else:
            return f'{type(self).__name__} - {self.name}'

    def __init__(self, node: Node, name=None):
        self.name = name

        if not node.node_type == NodeType.CPTNode:
            raise Exception(f'Only node of type {NodeType.CPTNode} supported')

        self.output_messages: Dict[NodeBase, Message] = {}
        self.input_messages: Dict[Message, NodeBase] = {}

    def add_output_message(self, destination, message):
        self.output_messages[destination] = message

    def add_input_message(self, input_message, factor_node):
        self.input_messages[input_message] = factor_node


class VariableNode(NodeBase):
    pass


class FactorNode(NodeBase):
    pass
