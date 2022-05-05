import string
from abc import abstractmethod
from typing import Union, List, Dict

import torch

from common.utilities import Cfg
from model.bayesian_network import BayesianNetwork
from model.nodes import Node, NodeType


class FactorGraph:
    def __init__(self,
                 bayesian_network: BayesianNetwork,
                 observed_nodes: List[Node],
                 device: torch.device,
                 num_observations: int):
        if not all([node.node_type == NodeType.CPTNode for node in bayesian_network.nodes]):
            raise Exception(f'Only nodes of type {NodeType.CPTNode} supported')

        self.device = device
        self.num_observations = num_observations
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
                message = Message(
                    variable_node,
                    self.factor_nodes[child],
                    torch.ones((self.num_observations, node.numK), dtype=torch.float64, device=self.device))
                variable_node.add_output_message(message)

            # To corresponding factor node
            message = Message(
                variable_node,
                factor_node,
                torch.ones((self.num_observations, node.numK), dtype=torch.float64, device=self.device))
            variable_node.add_output_message(message)

            ### Factor nodes
            # To children
            for parent in bayesian_network.parents[node]:
                message = Message(
                    factor_node,
                    self.variable_nodes[parent],
                    torch.ones((self.num_observations, parent.numK), dtype=torch.float64, device=self.device))
                factor_node.add_output_message(message)

            # To corresponding variable node
            message = Message(
                factor_node,
                variable_node,
                torch.ones((self.num_observations, node.numK), dtype=torch.float64, device=self.device))
            factor_node.add_output_message(message)

        # Bias inputs for leaf nodes
        for leaf_node in [node for node in bayesian_network.nodes if bayesian_network.is_leaf_node(node)]:
            bias_message = Message(
                source=None,
                destination=self.variable_nodes[leaf_node],
                initial_value=torch.ones((self.num_observations, leaf_node.numK), dtype=torch.float64, device=self.device))

            self.variable_nodes[leaf_node].add_fixed_input_message(bias_message)

        # Add input messages for observed nodes
        for observed_node in observed_nodes:
            input_message = Message(
                source=None,
                destination=self.variable_nodes[observed_node],
                initial_value=torch.ones((self.num_observations, observed_node.numK), dtype=torch.float64, device=self.device))

            self.variable_nodes[observed_node].add_fixed_input_message(input_message)

            self.observed_nodes_input_messages.append(input_message)

        # Configure root factor nodes to broadcast to number of observations
        for root_node in bayesian_network.root_nodes:
            self.factor_nodes[root_node].set_broadcast(self.num_observations)

        # Update internal registries
        for variable_node in self.variable_nodes.values():
            variable_node.configure_input_messages()

        for factor_node in self.factor_nodes.values():
            factor_node.configure_input_messages()

    def enter_evidence(self, evidence: List[torch.Tensor]):
        # evidence:
        # - len(evidence) = number of observed nodes
        # - torch.Tensor.shape = [number of trials, number of states]
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
    def __init__(self, device: torch.device, node: Node, name=None):
        if not node.node_type == NodeType.CPTNode:
            raise Exception(f'Only node of type {NodeType.CPTNode} supported')

        super().__init__(name)

        self.cpt = torch.tensor(node.cpt, dtype=torch.float64, device=device)
        self.num_observations = None

    def calculate_output_values(self):
        for (i, output_message) in enumerate(self.output_messages):
            # Collect input tensors for output message currently being calculated
            input_tensors = [
                input_message.get_value()
                for input_message
                in self.input_messages
                if input_message.source is not output_message.destination
            ]

            # Construct einsum equation
            all_indices = range(len(self.input_messages))
            current_indices = list(all_indices)
            del current_indices[i]

            equation = []

            for j, input_tensor in enumerate(input_tensors):
                equation.append(input_tensor)
                equation.append([..., current_indices[j]])

            all_inputs_symbol = [..., *all_indices]
            equation.append(self.cpt)
            equation.append(all_inputs_symbol)

            equation.append([..., i])

            # Perform calculation using einsum
            result = torch.einsum(*equation)

            # If root node, manually broadcast to number of observations
            if self.num_observations:
                result = result.repeat((self.num_observations, 1))

            # Set new value for output message
            output_message.set_new_value(result)

    def set_broadcast(self, num_observations: int):
        self.num_observations = num_observations

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
