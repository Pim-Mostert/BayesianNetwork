from abc import abstractmethod
from typing import Dict, List, Union

import torch

from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.common.torch_settings import TorchSettings


class FactorGraph:
    def __init__(
        self,
        bayesian_network: BayesianNetwork,
        observed_nodes: List[Node],
        torch_settings: TorchSettings,
        num_observations: int,
    ):
        self.torch_settings = torch_settings
        self.num_observations = num_observations if num_observations > 0 else 1
        self.observed_nodes = observed_nodes

        # Instantiate nodes
        self.factor_nodes: Dict[Node, FactorNode] = {
            node: FactorNode(
                node.cpt,
                self.num_observations,
                self.torch_settings,
                name=node.name,
            )
            for node in bayesian_network.nodes
        }
        self.variable_nodes: Dict[Node, VariableNode] = {
            node: VariableNode(
                self.num_observations,
                node.num_states,
                corresponding_factor_node=self.factor_nodes[node],
                is_leaf_node=bayesian_network.is_leaf_node(node),
                is_observed=node in observed_nodes,
                torch_settings=self.torch_settings,
                name=node.name,
            )
            for node in bayesian_network.nodes
        }

        # Create output messages
        for node in bayesian_network.nodes:
            variable_node = self.variable_nodes[node]
            factor_node = self.factor_nodes[node]

            ### Variable node
            # To children's factor nodes
            for child in bayesian_network.children[node]:
                variable_node.add_output_message(destination=self.factor_nodes[child])

            # To corresponding factor node
            variable_node.add_output_message(factor_node)

            ### Factor node
            # To parents' variable nodes
            for parent in bayesian_network.parents[node]:
                factor_node.add_output_message(self.variable_nodes[parent])

            # To corresponding variable node
            factor_node.add_output_message(variable_node)

        # Configure nodes' input messages
        all_factor_graph_nodes: List[FactorGraphNodeBase] = [
            *self.factor_nodes.values(),
            *self.variable_nodes.values(),
        ]
        for factor_graph_node in all_factor_graph_nodes:
            factor_graph_node.discover_input_messages()

    def enter_evidence(self, evidence: List[torch.Tensor]):
        # evidence:
        # - len(evidence) = number of observed nodes
        # - torch.Tensor.shape = [number of trials, number of states]
        for i, observed_node in enumerate(self.observed_nodes):
            variable_node = self.variable_nodes[observed_node]
            variable_node.observation_message.value = evidence[i]

    def iterate(self):
        # First evaluate factor nodes
        for factor_node in self.factor_nodes.values():
            factor_node.calculate_output_values()

        # Then variable nodes
        for variable_node in self.variable_nodes.values():
            variable_node.calculate_output_values()


class FactorGraphNodeBase:
    def __repr__(self):
        return super().__repr__() if self.name is None else f"{type(self).__name__} - {self.name}"

    def __init__(
        self,
        num_observations: int,
        num_states: int,
        torch_settings: TorchSettings,
        name=None,
    ):
        self.name = name
        self.num_observations = num_observations
        self.num_states = num_states
        self.torch_settings = torch_settings

        self.output_messages: List[Message] = []
        self.input_messages: List[Message] = []

        self.inputs_for_output: Dict[Message, List[Message]] = {}

    def _add_output_message(self, destination: "FactorGraphNodeBase"):
        initial_value = (
            torch.ones(
                (self.num_observations, self.num_states),
                dtype=self.torch_settings.dtype,
                device=self.torch_settings.device,
            )
            / self.num_states
        )

        output_message = Message(source=self, destination=destination, initial_value=initial_value)

        self.output_messages.append(output_message)

    def discover_input_messages(self):
        for output_message in self.output_messages:
            [corresponding_input_message] = [
                other_output_message
                for other_output_message in output_message.destination.output_messages
                if other_output_message.destination is self
            ]

            self.input_messages.append(corresponding_input_message)

        for output_message in self.output_messages:
            corresponding_input_messages = [
                input_message
                for input_message in self.input_messages
                if input_message.source is not output_message.destination
            ]

            self.inputs_for_output[output_message] = corresponding_input_messages

    @abstractmethod
    def calculate_output_values(self):
        raise NotImplementedError()


class VariableNode(FactorGraphNodeBase):
    def __init__(
        self,
        num_observations: int,
        num_states: int,
        corresponding_factor_node: "FactorNode",
        is_leaf_node: bool,
        is_observed: bool,
        torch_settings: TorchSettings,
        name=None,
    ):
        super().__init__(
            num_observations,
            num_states,
            torch_settings=torch_settings,
            name=name,
        )

        self.factor_node = corresponding_factor_node
        self.local_likelihood: torch.Tensor = torch.nan
        self.is_leaf_node = is_leaf_node
        self.is_observed = is_observed
        self.observation_message: Union[Message, None] = (
            Message(
                source=None,
                destination=self,
                initial_value=torch.ones(
                    (self.num_observations, self.num_states),
                    dtype=self.torch_settings.dtype,
                    device=self.torch_settings.device,
                ),
            )
            if self.is_observed
            else None
        )

    def discover_input_messages(self):
        if self.is_observed:
            self.input_messages.append(self.observation_message)

        super().discover_input_messages()

    def add_output_message(self, destination: "FactorNode"):
        self._add_output_message(destination)

    def calculate_output_values(self):
        # all_input_tensors: [num_observations x num_inputs x num_states]
        all_input_tensors = torch.stack(
            [input_message.value for input_message in self.input_messages], dim=1
        )

        # local_likelihood: [num_observations]
        self.local_likelihood = all_input_tensors.prod(dim=1).sum(axis=1)

        if self.is_leaf_node and not self.is_observed:
            [output_message] = self.output_messages
            output_message.value = (
                torch.ones(
                    (self.num_observations, self.num_states),
                    dtype=self.torch_settings.dtype,
                    device=self.torch_settings.device,
                )
                / self.local_likelihood[:, None]
            )
        else:
            for output_message in self.output_messages:
                # input_tensors: [num_observations x num_inputs x num_states]
                input_tensors = torch.stack(
                    [
                        input_message.value
                        for input_message in self.inputs_for_output[output_message]
                    ],
                    dim=1,
                )

                # [num_observations x num_states]
                result = input_tensors.prod(dim=1)

                if output_message.destination is self.factor_node:
                    result /= self.local_likelihood[:, None]
                else:
                    result /= result.sum(axis=1, keepdim=True)

                output_message.value = result


class FactorNode(FactorGraphNodeBase):
    def __init__(
        self,
        cpt: torch.Tensor,
        num_observations: int,
        torch_settings: TorchSettings,
        name=None,
    ):
        super().__init__(
            num_observations,
            num_states=cpt.shape[-1],
            torch_settings=torch_settings,
            name=name,
        )

        self.cpt = cpt
        self.is_root_node = len(cpt.shape) == 1

    def add_output_message(self, destination: "VariableNode"):
        self._add_output_message(destination)

    def calculate_output_values(self):
        for i, output_message in enumerate(self.output_messages):
            input_tensors = [
                input_message.value for input_message in self.inputs_for_output[output_message]
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
            if self.is_root_node:
                result = result.repeat((self.num_observations, 1))

            # Set new value for output message
            output_message.value = result


class Message:
    def __repr__(self):
        return f"{self.source} -> {self.destination}"

    def __init__(
        self,
        source: Union[FactorGraphNodeBase, None],
        destination: FactorGraphNodeBase,
        initial_value: torch.Tensor,
    ):
        self.source = source
        self.destination = destination
        self.value: torch.Tensor = initial_value
