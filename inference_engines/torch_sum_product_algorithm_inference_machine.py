import string
from abc import abstractmethod
from typing import List, Dict, Union

import torch

from common.utilities import Cfg
from model.bayesian_network import BayesianNetwork
from model.interfaces import IInferenceMachine
from model.nodes import Node, NodeType


class TorchSumProductAlgorithmInferenceMachine(IInferenceMachine):
    def __init__(self, cfg: Cfg, bayesian_network: BayesianNetwork, observed_nodes: List[Node]):
        self.bayesian_network = bayesian_network
        self.factor_graph = FactorGraph(cfg, bayesian_network)
        self.num_iterations = cfg.num_iterations
        self.callback = cfg.callback

    def infer(self, nodes: List[Node]) -> torch.Tensor:
        for _ in range(self.num_iterations):
            self.factor_graph.iterate()

            self.callback(self.factor_graph)

        if len(nodes) > 2:
            raise Exception("Only inference on single nodes or two neighbouring nodes supported")

        if len(nodes) == 1:
            return self._infer_single_node(nodes[0])
        else:
            return self._infer_neighbouring_nodes(nodes)

    def _infer_single_node(self, node: Node) -> torch.Tensor:
        variable_node = self.factor_graph.variable_nodes[node]
        factor_node = self.factor_graph.factor_nodes[node]

        [value_to_factor_node] = [
            message.get_value()
            for message
            in variable_node.output_messages
            if message.destination is factor_node
        ]
        [value_from_factor_node] = [
            message.get_value()
            for message
            in variable_node.input_messages
            if message.source is factor_node
        ]

        p = value_from_factor_node * value_to_factor_node

        p /= p.sum()

        return p

    def _infer_neighbouring_nodes(self, nodes: List[Node]) -> torch.Tensor:
        if len(nodes) == 2 and not self.bayesian_network.are_neighbours(nodes[0], nodes[1]):
            raise Exception("Only inference on single nodes or two neighbouring nodes supported")

        raise Exception("todo")

    def enter_evidence(self, evidence):
        pass



class Message:
    def __repr__(self):
        return f'{self.source} -> {self.destination}'

    def __init__(
            self,
            source: Union['FactorGraphNodeBase', None],
            destination: 'FactorGraphNodeBase',
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


class FactorGraphNodeBase:
    def __repr__(self):
        return super().__repr__() \
            if self.name is None \
            else f'{type(self).__name__} - {self.name}'

    def __init__(self, name=None):
        self.name = name

        self.output_messages: List[Message] = []
        self.input_messages: List[Message] = []

    def add_output_message(self, message: Message):
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

    def add_bias_input_message(self, message: Message):
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


class FactorGraph:
    def __init__(self, cfg: Cfg, bayesian_network: BayesianNetwork):
        if not all([node.node_type == NodeType.CPTNode for node in bayesian_network.nodes]):
            raise Exception(f'Only nodes of type {NodeType.CPTNode} supported')

        self.device = cfg.device

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

            self.variable_nodes[leaf_node].add_bias_input_message(bias_message)

        # Connect input messages
        # for node in bayesian_network.nodes:
        #     variable_node = self.variable_nodes[node]
        #     factor_node = self.factor_nodes[node]
        #
        #     # Variable nodes
        #     for child in bayesian_network.children[node]:
        #         input_message = self.factor_nodes[child].outputs[variable_node]
        #         variable_node.add_input_message(input_message)
        #
        #     input_message = factor_node.outputs[variable_node]
        #     variable_node.add_input_message(input_message)
        #
        #     if bayesian_network.is_leaf_node(node):
        #         input_message = Message(torch.ones((node.numK), dtype=torch.float64, device=self.device))
        #         variable_node.add_input_message(input_message)
        #
        #     # Factor nodes
        #     for parent in bayesian_network.parents[node]:
        #         input_message = self.variable_nodes[parent].outputs[factor_node]
        #         factor_node.add_input_message(input_message)
        #
        #     input_message = variable_node.outputs[factor_node]
        #     factor_node.add_input_message(input_message)
        #
        #     if bayesian_network.is_root_node(node):
        #         input_message = Message(torch.ones((node.numK), dtype=torch.float64, device=self.device))
        #         variable_node.add_input_message(input_message)

        # Update internal registries
        for variable_node in self.variable_nodes.values():
            variable_node.configure_input_messages()

        for factor_node in self.factor_nodes.values():
            factor_node.configure_input_messages()

    def iterate(self):
        for variable_node in self.variable_nodes.values():
            variable_node.calculate_output_values()

        for factor_node in self.factor_nodes.values():
            factor_node.calculate_output_values()

        for variable_node in self.variable_nodes.values():
            variable_node.flip()

        for factor_node in self.factor_nodes.values():
            factor_node.flip()

