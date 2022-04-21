import string
from abc import abstractmethod
from typing import List, Dict

import torch

from common.utilities import Cfg
from model.bayesian_network import BayesianNetwork
from model.interfaces import IInferenceMachine
from model.nodes import Node, NodeType


class TorchSumProductAlgorithmInferenceMachine(IInferenceMachine):
    def __init__(self, cfg: Cfg, bayesian_network: BayesianNetwork, observed_nodes: List[Node]):
        self.factor_graph = FactorGraph(cfg, bayesian_network)
        self.num_iterations = cfg.num_iterations

    def infer(self, nodes: List[Node]):
        for _ in range(self.num_iterations):
            self.factor_graph.iterate()

    def enter_evidence(self, evidence):
        pass


class Message:
    def __init__(self, initial_value: torch.Tensor):
        self.value: torch.Tensor = initial_value
        self.new_value: torch.Tensor = initial_value

    def flip(self):
        self.value = self.new_value

    def set_new_value(self, value: torch.Tensor):
        self.new_value = value

    def get_value(self) -> torch.Tensor:
        return self.value


class FactorGraphNodeBase:
    def __repr__(self):
        return super().__repr__() \
            if self.name is None \
            else f'{type(self).__name__} - {self.name}'

    def __init__(self, name=None):
        self.name = name

        self.outputs: Dict[FactorGraphNodeBase, Message] = {}
        self.inputs: List[Message] = []
        self.output_parents: Dict[Message, List[Message]] = {}

    def add_output_message(self, message: Message, destination: 'FactorGraphNodeBase'):
        self.outputs[destination] = message

    def add_input_message(self, input_message: Message):
        self.inputs.append(input_message)

    def update_links(self):
        for output_destination in self.outputs:
            other_output_messages = [
                self.outputs[other_output_destination]
                for other_output_destination
                in self.outputs
                if output_destination is not other_output_destination]

            output_message = self.outputs[output_destination]
            self.output_parents[output_message] = other_output_messages

    def flip(self):
        for output_message in self.outputs.values():
            output_message.flip()

    @abstractmethod
    def calculate_output_values(self):
        pass


class VariableNode(FactorGraphNodeBase):
    def calculate_output_values(self):
        for output_message in self.outputs.values():
            input_messages = self.output_parents[output_message]
            input_tensors = [input_message.get_value() for input_message in input_messages]

            result = torch.stack(input_tensors).prod(dim=0)

            output_message.set_new_value(result)


class FactorNode(FactorGraphNodeBase):
    def __init__(self, node: Node, name=None):
        if not node.node_type == NodeType.CPTNode:
            raise Exception(f'Only node of type {NodeType.CPTNode} supported')

        super().__init__(name)

        self.node = node
        self.einsum_equation: str = ''

    def update_links(self):
        super().update_links()

        num_inputs = len(self.inputs)
        letters = string.ascii_letters
        if num_inputs > len(letters):
            raise Exception(f'Max {len(letters)} inputs supported at this moment')

        letters_subset = letters[0:num_inputs]
        self.einsum_equation += ','.join(letters_subset)
        self.einsum_equation += ','
        self.einsum_equation += letters_subset
        self.einsum_equation += '->'
        self.einsum_equation += letters_subset

    def calculate_output_values(self):
        input_tensors = [input.get_value() for input in self.inputs]
        p = torch.einsum(self.einsum_equation, *input_tensors, self.node.cpt)

        all_dims = list(range(len(p.shape)))
        for i, output in enumerate(self.outputs):
            dims = list(all_dims)
            del dims[i]
            output_value = p.sum(dim=dims)

            output_message = self.outputs[output]
            output_message.set_new_value(output_value)


class FactorGraph:
    def __init__(self, cfg: Cfg, bayesian_network: BayesianNetwork):
        if not all([node.node_type == NodeType.CPTNode for node in bayesian_network.nodes]):
            raise Exception(f'Only nodes of type {NodeType.CPTNode} supported')

        self.device = cfg.device

        # Instantiate nodes
        self.variable_nodes: Dict[Node, VariableNode] = {node: VariableNode(name=node.name) for node in bayesian_network.nodes}
        self.factor_nodes: Dict[Node, FactorNode] = {node: FactorNode(node, name=node.name) for node in bayesian_network.nodes}

        # Output messages
        for node in bayesian_network.nodes:
            variable_node = self.variable_nodes[node]
            factor_node = self.factor_nodes[node]

            ### Variable nodes
            # To corresponding factor node
            message = Message(torch.rand((node.numK), dtype=torch.float64, device=self.device))
            variable_node.add_output_message(message, factor_node)

            # To children
            for child in bayesian_network.children[node]:
                message = Message(torch.rand((node.numK), dtype=torch.float64, device=self.device))
                variable_node.add_output_message(message, self.factor_nodes[child])

            ### Factor nodes
            # To corresponding variable node
            message = Message(torch.rand((node.numK), dtype=torch.float64, device=self.device))
            factor_node.add_output_message(message, variable_node)

            # To children
            for parent in bayesian_network.parents[node]:
                message = Message(torch.rand((parent.numK), dtype=torch.float64, device=self.device))
                factor_node.add_output_message(message, self.variable_nodes[parent])

        # Connect input messages
        for node in bayesian_network.nodes:
            variable_node = self.variable_nodes[node]
            factor_node = self.factor_nodes[node]

            # Variable nodes
            for child in bayesian_network.children[node]:
                input_message = self.factor_nodes[child].outputs[variable_node]
                variable_node.add_input_message(input_message)

            input_message = factor_node.outputs[variable_node]
            variable_node.add_input_message(input_message)

            if bayesian_network.is_leaf_node(node):
                input_message = Message(torch.ones((node.numK), dtype=torch.float64, device=self.device))
                variable_node.add_input_message(input_message)

            # Factor nodes
            for parent in bayesian_network.parents[node]:
                input_message = self.variable_nodes[parent].outputs[factor_node]
                factor_node.add_input_message(input_message)

            input_message = variable_node.outputs[factor_node]
            factor_node.add_input_message(input_message)

            if bayesian_network.is_root_node(node):
                input_message = Message(torch.ones((node.numK), dtype=torch.float64, device=self.device))
                variable_node.add_input_message(input_message)

        # Update internal registries
        for variable_node in self.variable_nodes.values():
            variable_node.update_links()

        for factor_node in self.factor_nodes.values():
            factor_node.update_links()

    def iterate(self):
        for variable_node in self.variable_nodes.values():
            variable_node.calculate_output_values()

        for factor_node in self.factor_nodes.values():
            factor_node.calculate_output_values()

        for variable_node in self.variable_nodes.values():
            variable_node.flip()

        for factor_node in self.factor_nodes.values():
            factor_node.flip()

