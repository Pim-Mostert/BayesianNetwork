from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch

from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.common.torch_settings import TorchSettings


class FactorGraphNodeBase(ABC):
    def __init__(
        self,
        torch_settings: TorchSettings,
        name: Optional[str] = None,
    ):
        self.torch_settings = torch_settings
        self.name = name

    def __repr__(self):
        return super().__repr__() if self.name is None else f"{type(self).__name__} - {self.name}"

    @abstractmethod
    def calculate_outputs(self) -> None:
        raise NotImplementedError()


class VariableNode(FactorGraphNodeBase):
    class OutputWithInputIndices:
        def __init__(self, output: torch.Tensor, input_indices: List[int]):
            self.output = output
            self.input_indices = input_indices

    def __init__(
        self,
        torch_settings: TorchSettings,
        num_observations: int,
        num_inputs: int,
        num_states: int,
        local_likelihood: torch.Tensor,
        is_observed: bool,
        name: Optional[str] = None,
    ):
        super().__init__(torch_settings, name)

        # To handle calculation without any evidence
        if num_observations == 0:
            num_observations = 1

        self.local_likelihood = local_likelihood
        self.is_observed = is_observed

        if self.is_observed:
            self._all_inputs: torch.Tensor = (
                torch.ones(
                    (num_inputs + 1, num_observations, num_states),
                    dtype=self.torch_settings.dtype,
                    device=self.torch_settings.device,
                )
                / num_states
            )
            self.input_from_local_factor_node: torch.Tensor = self._all_inputs[-1]
            self.input_from_observation: torch.Tensor = self._all_inputs[-2]
            self.input_from_remote_factor_nodes: torch.Tensor = self._all_inputs[:-2]
        else:
            self._all_inputs: torch.Tensor = (
                torch.ones(
                    (num_inputs, num_observations, num_states),
                    dtype=self.torch_settings.dtype,
                    device=self.torch_settings.device,
                )
                / num_states
            )
            self.input_from_local_factor_node: torch.Tensor = self._all_inputs[-1]
            self.input_from_remote_factor_nodes: torch.Tensor = self._all_inputs[:-1]

        self.output_with_indices_to_local_factor_node: VariableNode.OutputWithInputIndices = (
            VariableNode.OutputWithInputIndices(
                torch.empty(()),
                [d for d in range(len(self._all_inputs) - 1)],  # Placeholder
            )
        )
        self.outputs_with_indices_to_remote_factor_nodes: List[
            VariableNode.OutputWithInputIndices
        ] = [
            VariableNode.OutputWithInputIndices(
                torch.empty(()),
                [d for d in range(len(self._all_inputs)) if d != i],  # Placeholder
            )
            for i in range(num_inputs - 1)
        ]

    def set_observations(self, observations: torch.Tensor) -> None:
        if not self.is_observed:
            raise Exception(
                "Can't set observation for variable nodes that are not marked as observed"
            )

        self.input_from_observation[:] = observations

    def calculate_outputs(self) -> None:
        # Output to local factor node
        local_output = self.output_with_indices_to_local_factor_node.output
        indices = self.output_with_indices_to_local_factor_node.input_indices

        c = self._all_inputs.prod(dim=0, keepdim=True).sum(dim=2, keepdim=True)
        local_output[:] = self._all_inputs[indices].prod(axis=0) / c

        # Store local_likelihood for log-likelihood calculation
        self.local_likelihood[:] = c.squeeze()

        for remote_output_with_input_indices in self.outputs_with_indices_to_remote_factor_nodes:
            remote_output = remote_output_with_input_indices.output
            indices = remote_output_with_input_indices.input_indices

            remote_output[:] = self._all_inputs[indices].prod(dim=0)
            remote_output /= remote_output.sum(dim=1, keepdim=True)


class FactorNode(FactorGraphNodeBase):
    def __init__(
        self,
        torch_settings: TorchSettings,
        num_observations: int,
        num_inputs: int,
        inputs_num_states: List[int],
        cpt: torch.Tensor,
        name: Optional[str] = None,
    ):
        super().__init__(torch_settings, name)

        # To handle calculation without any evidence
        if num_observations == 0:
            num_observations = 1

        self.all_inputs: List[torch.Tensor] = [
            torch.ones(
                (num_observations, input_num_states),
                dtype=self.torch_settings.dtype,
                device=self.torch_settings.device,
            )
            / input_num_states
            for input_num_states in inputs_num_states
        ]
        self.inputs_from_remote_variable_nodes = self.all_inputs[:-1]
        self.input_from_local_variable_node = self.all_inputs[-1]
        self._all_outputs: List[torch.Tensor] = [
            torch.empty(())
            for _ in range(num_inputs)  # Placeholder
        ]
        self.output_to_local_variable_node = self._all_outputs[-1]
        self.outputs_to_remote_variable_nodes = self._all_outputs[:-1]
        self.cpt = cpt

        self.einsum_equations: List[List] = [
            self._construct_einsum_equations_for_output(output_index)
            for output_index, _ in enumerate(self._all_outputs)
        ]

    def configure_output_to_local_variable_node(self, input):
        self.configure_output_to_remote_variable_node(-1, input)

    def configure_output_to_remote_variable_node(self, output_index, input):
        self._all_outputs[output_index] = input

    def _construct_einsum_equations_for_output(self, output_index: int) -> List:
        all_indices = range(len(self._all_outputs))

        # Get all inputs with indices, except for the input corresponding to current output
        inputs_with_indices = (
            (input, index)
            for input, index in zip(self.all_inputs, all_indices)
            if index != output_index
        )

        # Example einsum equation:
        #   'na, nc, nd, abcd->nb', input0, input2, input3, cpt
        #   [input0, [..., 0], input2, [..., 2], input3, [..., 3], cpt, [..., 0, 1, 2, 3], [..., 1]]
        einsum_equation = []

        # Each input used to calculate current output
        for input, index in inputs_with_indices:
            einsum_equation.append(input)
            einsum_equation.append([..., index])

        # Cpt of the factor node
        einsum_equation.append(self.cpt)
        einsum_equation.append([..., *all_indices])

        # Desired output dimensions
        einsum_equation.append([..., output_index])

        return einsum_equation

    def calculate_outputs(self) -> None:
        for output, einsum_equation in zip(self._all_outputs, self.einsum_equations):
            output[:] = torch.einsum(*einsum_equation)


class FactorGraph:
    def __init__(
        self,
        bayesian_network: BayesianNetwork,
        num_observations: int,
        observed_nodes: List[Node],
        torch_settings: TorchSettings,
    ):
        self.torch_settings = torch_settings
        self.observed_nodes = observed_nodes
        self.local_likelihoods: torch.Tensor = torch.zeros(
            (num_observations, len(bayesian_network.nodes)),
            dtype=self.torch_settings.dtype,
            device=self.torch_settings.device,
        )

        # Instantiate nodes
        self.variable_nodes: Dict[Node, VariableNode] = {
            node: VariableNode(
                num_observations=num_observations,
                num_inputs=len(list(bayesian_network.children_of(node))) + 1,
                num_states=node.num_states,
                local_likelihood=self.local_likelihoods[:, i],
                is_observed=node in observed_nodes,
                torch_settings=self.torch_settings,
                name=node.name,
            )
            for i, node in enumerate(bayesian_network.nodes)
        }

        self.factor_nodes: Dict[Node, FactorNode] = {
            node: FactorNode(
                num_observations=num_observations,
                num_inputs=len(list(bayesian_network.parents_of(node))) + 1,
                inputs_num_states=[
                    parent.num_states for parent in bayesian_network.parents_of(node)
                ]
                + [node.num_states],
                cpt=node.cpt,
                torch_settings=self.torch_settings,
                name=node.name,
            )
            for node in bayesian_network.nodes
        }

        for node in bayesian_network.nodes:
            variable_node = self.variable_nodes[node]
            factor_node = self.factor_nodes[node]

            # ### Variable nodes
            # Corresponding factor node
            variable_node.output_with_indices_to_local_factor_node.output = (
                factor_node.input_from_local_variable_node
            )

            # Child factor nodes
            child_nodes = bayesian_network.children_of(node)
            for child_node, outputs_with_input_indices in zip(
                child_nodes, variable_node.outputs_with_indices_to_remote_factor_nodes
            ):
                child_factor_node = self.factor_nodes[child_node]
                child_input_index = list(bayesian_network.parents_of(child_node)).index(node)
                outputs_with_input_indices.output = (
                    child_factor_node.inputs_from_remote_variable_nodes[child_input_index]
                )

            # ### Factor nodes
            # Parent variable nodes
            parent_nodes = bayesian_network.parents_of(node)
            for i, parent_node in enumerate(parent_nodes):
                parent_variable_node = self.variable_nodes[parent_node]
                child_index = list(bayesian_network.children_of(parent_node)).index(node)
                input_from_parent_variable_node = (
                    parent_variable_node.input_from_remote_factor_nodes[child_index]
                )
                factor_node.configure_output_to_remote_variable_node(
                    i, input_from_parent_variable_node
                )

            # Corresponding variable node
            factor_node.configure_output_to_local_variable_node(
                variable_node.input_from_local_factor_node
            )

    def iterate(self) -> None:
        for variable_node in self.variable_nodes.values():
            variable_node.calculate_outputs()

        for factor_node in self.factor_nodes.values():
            factor_node.calculate_outputs()

    def enter_evidence(self, evidence):
        for observed_node, observations in zip(self.observed_nodes, evidence):
            variable_node = self.variable_nodes[observed_node]
            variable_node.set_observations(observations)
