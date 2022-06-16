from abc import abstractmethod, ABC
from typing import Union, List, Dict, Optional

import torch

from bayesian_network.bayesian_network import BayesianNetwork, Node


class FactorGraphNodeBase(ABC):
    def __init__(self, name: Optional[str] = None):
        self.name = name

    def __repr__(self):
        return super().__repr__() \
            if self.name is None \
            else f'{type(self).__name__} - {self.name}'

    @abstractmethod
    def calculate_outputs(self) -> None:
        pass


class VariableNode(FactorGraphNodeBase):
    class OutputWithInputIndices:
        def __init__(self, output: torch.Tensor, input_indices: List[int]):
            self.output = output
            self.input_indices = input_indices

    def __init__(self,
                 device: torch.device,
                 num_observations: int,
                 num_inputs: int,
                 num_states: int,
                 c: torch.Tensor,
                 is_observed: bool,
                 name: Optional[str] = None):
        super().__init__(name)

        # To handle calculation without any evidence
        if num_observations == 0:
            num_observations = 1

        self.inputs = \
            torch.ones((num_inputs, num_observations, num_states), dtype=torch.double, device=device) / num_states \
            if not is_observed \
            else torch.ones((num_inputs+1, num_observations, num_states), dtype=torch.double, device=device) / num_states
        self.local_output: torch.Tensor = torch.empty(())      # Placeholder
        self.outputs_with_input_indices: List[VariableNode.OutputWithInputIndices] = [
            VariableNode.OutputWithInputIndices(
                torch.empty(()),     # Placeholder for output tensor
                [d for d in range(len(self.inputs)) if d != i]
            )
            for i
            in range(num_inputs-1)
        ]
        self.c = c

    def set_observations(self, observations: torch.Tensor) -> None:
        self.inputs[-2] = observations

    def calculate_outputs(self) -> None:
        c = self.inputs.prod(dim=0, keepdim=True).sum(dim=2, keepdim=True)
        self.local_output[:] = self.inputs[:-1].prod(axis=0) / c
        self.c[:] = c.squeeze()

        for output_with_input_indices in self.outputs_with_input_indices:
            output = output_with_input_indices.output
            input_indices = output_with_input_indices.input_indices

            output[:] = self.inputs[input_indices].prod(dim=0)
            output /= output.sum(dim=1, keepdim=True)


class FactorNode(FactorGraphNodeBase):
    def __init__(self,
                 device: torch.device,
                 num_observations: int,
                 num_inputs: int,
                 num_input_states: List[int],
                 cpt: torch.Tensor,
                 name: Optional[str] = None):
        super().__init__(name)

        # To handle calculation without any evidence
        if num_observations == 0:
            num_observations = 1

        self.inputs: List[torch.Tensor] = [
            torch.ones((num_observations, num_states), dtype=torch.double, device=device) / num_states
            for num_states
            in num_input_states
        ]
        self.outputs: List[Union[torch.Tensor, None]] = [None] * num_inputs
        self.cpt = cpt

    def calculate_outputs(self) -> None:
        indices = range(len(self.outputs))

        for i, output in enumerate(self.outputs):
            indices_with_inputs = ((index, input) for index, input in zip(indices, self.inputs) if index != i)

            # Example einsum equation:
            #   'na, nc, nd, abcd->nb', input0, input2, input3, cpt
            #   [input0, [..., 0], input2, [..., 2], input3, [..., 3], cpt, [..., 0, 1, 2, 3], [..., 1]]
            def einsum_equation_generator():
                # Each input used to calculate current output
                for index, input in indices_with_inputs:
                    yield input
                    yield [..., index]

                # Cpt of the factor node
                yield self.cpt
                yield [..., *indices]

                # Desired output dimensions
                yield [..., i]

            output[:] = torch.einsum(*einsum_equation_generator())


class FactorGraph:
    def __init__(self,
                 bayesian_network: BayesianNetwork,
                 num_observations: int,
                 observed_nodes: List[Node],
                 device: torch.device):
        self.device = device
        self.observed_nodes = observed_nodes
        self.c: torch.Tensor = torch.zeros((num_observations, len(bayesian_network.nodes)), dtype=torch.double, device=self.device)

        # Instantiate nodes
        self.variable_nodes: Dict[Node, VariableNode] = {
            node: VariableNode(
                num_observations=num_observations,
                num_inputs=len(bayesian_network.children[node]) + 1,
                num_states=node.num_states,
                c=self.c[:, i],
                is_observed=node in observed_nodes,
                device=device,
                name=node.name)
            for i, node
            in enumerate(bayesian_network.nodes)
        }

        self.factor_nodes: Dict[Node, FactorNode] = {
            node: FactorNode(
                num_observations=num_observations,
                num_inputs=len(bayesian_network.parents[node]) + 1,
                num_input_states=[parent.num_states for parent in bayesian_network.parents[node]] + [node.num_states],
                cpt=node.cpt,
                device=device,
                name=node.name)
            for node
            in bayesian_network.nodes
        }

        for node in bayesian_network.nodes:
            variable_node = self.variable_nodes[node]
            factor_node = self.factor_nodes[node]

            ### Variable nodes
            # Corresponding factor node
            variable_node.local_output = factor_node.inputs[-1]

            # Child factor nodes
            child_nodes = bayesian_network.children[node]
            for i, child_node in enumerate(child_nodes):
                child_factor_node = self.factor_nodes[child_node]
                child_input_index = bayesian_network.parents[child_node].index(node)
                variable_node.outputs_with_input_indices[i].output = child_factor_node.inputs[child_input_index]

            ### Factor nodes
            # Parent variable nodes
            parent_nodes = bayesian_network.parents[node]
            for i, parent_node in enumerate(parent_nodes):
                parent_variable_node = self.variable_nodes[parent_node]
                parent_input_index = bayesian_network.children[parent_node].index(node)
                factor_node.outputs[i] = parent_variable_node.inputs[parent_input_index]

            # Corresponding variable node
            factor_node.outputs[-1] = variable_node.inputs[-1]

    def iterate(self) -> None:
        for variable_node in self.variable_nodes.values():
            variable_node.calculate_outputs()

        for factor_node in self.factor_nodes.values():
            factor_node.calculate_outputs()

    def enter_evidence(self, evidence):
        for observed_node, observations in zip(self.observed_nodes, evidence):
            variable_node = self.variable_nodes[observed_node]
            variable_node.inputs[-2] = observations
