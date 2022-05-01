import string
from abc import abstractmethod
from typing import Union, List, Dict

import torch

from common.utilities import Cfg
from model.bayesian_network import BayesianNetwork
from model.nodes import Node, NodeType


class FactorGraph:
    def __init__(self, cfg: Cfg, bayesian_network: BayesianNetwork, observed_nodes: List[Node]):
        if not all([node.node_type == NodeType.CPTNode for node in bayesian_network.nodes]):
            raise Exception(f'Only nodes of type {NodeType.CPTNode} supported')

        self.device = cfg.device
        self.observed_nodes_input_messages: List[Message] = []

        # Instantiate nodes
        self.variable_nodes: Dict[Node, VariableNode] = {node: VariableNode(name=node.name) for node in bayesian_network.nodes}
        self.factor_nodes: Dict[Node, FactorNode] = {node: FactorNode(self.device, node, name=node.name) for node in bayesian_network.nodes}

        # Output messages
        for node in bayesian_network.nodes:
            variable_node = self.variable_nodes[node]
            factor_node = self.factor_nodes[node]

            ### Variable nodes
            # To children
            for child in bayesian_network.children[node]:
                message = Message(variable_node, self.factor_nodes[child], torch.rand((node.numK), dtype=torch.float64, device=self.device))
                variable_node.add_output_message(message)

            # To corresponding factor node
            message = Message(variable_node, factor_node, torch.rand((node.numK), dtype=torch.float64, device=self.device))
            variable_node.add_output_message(message)

            ### Factor nodes
            # To children
            for parent in bayesian_network.parents[node]:
                message = Message(factor_node, self.variable_nodes[parent], torch.rand((parent.numK), dtype=torch.float64, device=self.device))
                factor_node.add_output_message(message)

            # To corresponding variable node
            message = Message(factor_node, variable_node, torch.rand((node.numK), dtype=torch.float64, device=self.device))
            factor_node.add_output_message(message)

        # Bias inputs for leaf nodes
        for leaf_node in [node for node in bayesian_network.nodes if bayesian_network.is_leaf_node(node)]:
            bias_message = Message(
                source=None,
                destination=self.variable_nodes[leaf_node],
                initial_value=torch.ones((leaf_node.numK), dtype=torch.float64, device=self.device))

            self.variable_nodes[leaf_node].add_fixed_input_message(bias_message)

        # Add input messages for observed nodes
        for observed_node in observed_nodes:
            input_message = Message(
                source=None,
                destination=self.variable_nodes[observed_node],
                initial_value=torch.ones((observed_node.numK), dtype=torch.float64, device=self.device))

            self.variable_nodes[observed_node].add_fixed_input_message(input_message)

            self.observed_nodes_input_messages.append(input_message)

        # Update internal registries
        for variable_node in self.variable_nodes.values():
            variable_node.configure_input_messages()

        for factor_node in self.factor_nodes.values():
            factor_node.configure_input_messages()

    def enter_evidence(self, evidence: List[torch.Tensor]):
        for i, input_message in enumerate(self.observed_nodes_input_messages):
            input_message.set_new_value(evidence[i])
            input_message.flip()

    def iterate(self):
        for variable_node in self.variable_nodes.values():
            variable_node.calculate_output_values()

        for factor_node in self.factor_nodes.values():
            factor_node.calculate_output_values()

        for variable_node in self.variable_nodes.values():
            variable_node.flip()

        for factor_node in self.factor_nodes.values():
            factor_node.flip()


class FactorGraphNodeBase:
    def __repr__(self):
        return super().__repr__() \
            if self.name is None \
            else f'{type(self).__name__} - {self.name}'

    def __init__(self, name=None):
        self.name = name

        self.output_messages: List[Message] = []
        self.input_messages: List[Message] = []

    def add_output_message(self, message: 'Message'):
        self.output_messages.append(message)

    def configure_input_messages(self):
        for output_message in self.output_messages:
            [corresponding_input_message] = [
                other_output_message for
                other_output_message
                in output_message.destination.output_messages
                if other_output_message.destination is self
            ]

            self.input_messages.append(corresponding_input_message)

    def flip(self):
        for output_message in self.output_messages:
            output_message.flip()

    @abstractmethod
    def calculate_output_values(self):
        pass


class VariableNode(FactorGraphNodeBase):
    def __init__(self, name=None):
        super().__init__(name=name)

        self.bias_messages: List[Message] = []

    def calculate_output_values(self):
        for output_message in self.output_messages:
            input_tensors = [
                input_message.get_value()
                for input_message
                in self.input_messages
                if input_message.source is not output_message.destination
            ]

            result = torch.stack(input_tensors).prod(dim=0)

            output_message.set_new_value(result)

    def add_fixed_input_message(self, message: 'Message'):
        self.bias_messages.append(message)

    def configure_input_messages(self):
        super().configure_input_messages()

        self.input_messages += self.bias_messages


class FactorNode(FactorGraphNodeBase):
    def __init__(self, device, node: Node, name=None):
        if not node.node_type == NodeType.CPTNode:
            raise Exception(f'Only node of type {NodeType.CPTNode} supported')

        super().__init__(name)

        self.cpt = torch.tensor(node.cpt, dtype=torch.float64, device=device)
        self.einsum_equation_start: str = ''
        self.equation_letters = None

    def configure_input_messages(self):
        super().configure_input_messages()

        num_inputs = len(self.input_messages)
        letters = string.ascii_letters
        if num_inputs > len(letters):
            raise Exception(f'Max {len(letters)} inputs supported at this moment')

        self.equation_letters = letters[0:num_inputs]

    def calculate_output_values(self):
        for (i, output_message) in enumerate(self.output_messages):
            input_tensors = [
                input_message.get_value()
                for input_message
                in self.input_messages
                if input_message.source is not output_message.destination
            ]

            equation_letters = list(self.equation_letters)
            current_letter = equation_letters[i]
            del equation_letters[i]

            equation = ','.join([*equation_letters, self.equation_letters])
            equation += '->'
            equation += current_letter

            result = torch.einsum(equation, [*input_tensors, self.cpt])

            output_message.set_new_value(result)


class Message:
    def __repr__(self):
        return f'{self.source} -> {self.destination}'

    def __init__(
            self,
            source: Union[FactorGraphNodeBase, None],
            destination: FactorGraphNodeBase,
            initial_value: torch.Tensor):
        self.source = source
        self.destination = destination
        self.value: torch.Tensor = initial_value
        self.new_value: torch.Tensor = initial_value

    def flip(self):
        self.value = self.new_value

    def set_new_value(self, value: torch.Tensor):
        self.new_value = value

    def get_value(self) -> torch.Tensor:
        return self.value
