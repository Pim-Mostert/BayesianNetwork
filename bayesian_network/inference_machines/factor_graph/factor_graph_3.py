from collections import namedtuple
import itertools
from typing import List, Dict
from varname import nameof

import torch

from bayesian_network.bayesian_network import BayesianNetwork, Node


class VariableNodeGroup:
    def __init__(self,
                 nodes: List[Node],
                 local_likelihoods: List[torch.Tensor],
                 children: Dict[Node, List[Node]],
                 parents: Dict[Node, List[Node]],
                 num_inputs: int,
                 num_states: int,
                 num_observations: int,
                 device: torch.device):
        self._device = device
        self.nodes = nodes
        self.local_likelihoods = local_likelihoods
        self._children = children
        self._parents = parents
        self._num_nodes = len(self.nodes)
        self._num_observations = num_observations
        self._num_inputs = num_inputs
        self._num_states = num_states
        self._num_outputs = self._num_inputs

        if self._num_observations == 0:
            self._num_observations = 1

        self._inputs = torch.ones(
            (
                self._num_inputs, 
                self._num_nodes, 
                self._num_observations,
                self._num_states
            ), device=self._device, dtype=torch.double) / num_states

        self._output_tensors = [
            [
                # Placeholder
                torch.zeros((self._num_observations, self._num_states), device=self._device, dtype=torch.double)
                for _
                in self.nodes
            ]
            for _
            in range(self._num_outputs)
        ]

        self._i_output_to_local_factor_node = self._num_outputs - 1

        self._calculation_output_tensor = torch.empty(())      # Placeholder
        self._calculation_result = torch.empty(())             # Placeholder

        # output_tensor[0][:], output_tensor[1][:], ..., output_tensor[self.num_nodes-1][:]
        self._calculation_assignment_statement =  \
            ', '.join([f'self.{nameof(self._calculation_output_tensor)}[{i_node}][:]' for i_node in range(self._num_nodes)]) \
                + f' = self.{nameof(self._calculation_result)}'

        self._calculation_indices_per_i_output = {
            i_output: [
                i_output_inner
                for i_output_inner
                in range(self._num_outputs)
                if i_output_inner != i_output
            ]
            for i_output
            in range(self._num_outputs)
        }

    def calculate_outputs(self):
        x = self._inputs.prod(axis=0)
        c = x.sum(axis=2)

        for local_likelihood, c_node in zip(self.local_likelihoods, c):
            local_likelihood[:] = c_node

        for i_output, output_tensors in enumerate(self._output_tensors):
            self._calculation_output_tensor = output_tensors

            self._calculation_result = x / self._inputs[i_output]
            
            if i_output == self._i_output_to_local_factor_node:
                self._calculation_result /= c[:, :, None]
            else:
                self._calculation_result /= self._calculation_result.sum(axis=2, keepdims=True)
                            
            exec(self._calculation_assignment_statement)

    def set_output_tensor(self, node: Node, output_node: Node, tensor: torch.Tensor):
        i_node = self.nodes.index(node)
        
        if output_node == node:
            i_output = self._i_output_to_local_factor_node
        else:
            i_output = self._children[node].index(output_node)

        self._output_tensors[i_output][i_node] = tensor

    def get_input_tensor(self, node: Node, input_node: Node) -> torch.Tensor:
        i_node = self.nodes.index(node)
        
        if input_node == node:
            i_input = self._i_output_to_local_factor_node
        else:
            i_input = self._children[node].index(input_node)

        return self._inputs[i_input, i_node]

    def set_observations(self, observed_node: Node, observations: torch.Tensor):
        i_node = self.nodes.index(observed_node)
        i_input = -2

        self._inputs[i_input, i_node] = observations


class FactorNodeGroup:
    def __init__(self,
                 nodes: List[Node],
                 children: Dict[Node, List[Node]],
                 parents: Dict[Node, List[Node]],
                 num_inputs: int,
                 inputs_num_states: List[int],
                 num_observations: int,
                 device: torch.device):
        self._device = device
        self.nodes = nodes
        self._children = children
        self._parents = parents
        self._num_nodes = len(self.nodes)
        self._num_observations = num_observations
        self._num_inputs = num_inputs
        self._inputs_num_states = inputs_num_states
        self._num_outputs = self._num_inputs
        self._outputs_num_states = self._inputs_num_states
        self._cpts = torch.stack([node.cpt for node in self.nodes])

        if self._num_observations == 0:
            self._num_observations = 1

        self._inputs = [
            torch.ones((
                self._num_nodes,
                self._num_observations,
                self._inputs_num_states[i_input]), device=self._device, dtype=torch.double) 
                    / torch.tensor(self._inputs_num_states[i_input], device=self._device, dtype=torch.double)
            for i_input
            in range(self._num_inputs)
        ]

        self.node_cpts = {
            node: self._cpts[i]
            for i, node
            in enumerate(self.nodes)
        }

        self._output_tensors = [
            [
                # Placeholder
                torch.zeros((self._num_observations, self._outputs_num_states[i_output]), device=self._device, dtype=torch.double)
                for _
                in range(self._num_nodes)
            ]
            for i_output
            in range(self._num_outputs)
        ]

        # self._output_tensor[0][:], self._output_tensor[1][:], ..., self._output_tensor[self.num_nodes-1][:]
        self._calculation_output_tensor = torch.empty(())      # Placeholder
        self._calculation_result = torch.empty(())             # Placeholder
        self._calculation_assignment_statement =  \
            ', '.join([f'self.{nameof(self._calculation_output_tensor)}[{i_node}][:]' for i_node in range(self._num_nodes)]) \
                + f' = self.{nameof(self._calculation_result)}'

        self._calculation_einsum_equation_per_output = [
            self._construct_einsum_equation_for_output(i_output)
            for i_output
            in range(self._num_outputs)
        ]


    def calculate_outputs(self):
        for i_output in range(self._num_outputs):
            self._calculation_output_tensor = self._output_tensors[i_output]
            
            einsum_equation = self._calculation_einsum_equation_per_output[i_output]
            self._calculation_result = torch.einsum(*einsum_equation)

            exec(self._calculation_assignment_statement)

    def _construct_einsum_equation_for_output(self, i_output: int) -> List:
        # Get all inputs with indices, except for the input corresponding to current output
        inputs_with_indices = [
            (input, index)
            for input, index
            in zip(self._inputs, range(self._num_inputs))
            if index != i_output
        ]

        # Example einsum equation:
        #   'kna, knc, knd, kabcd->knb', input0, input2, input3, cpt
        #   [input0, [..., 0], input2, [..., 2], input3, [..., 3], cpt, [..., 0, 1, 2, 3], [..., 1]]
        #
        # k: num_nodes
        # n: num_observations
        einsum_equation = []

        if not inputs_with_indices:
            einsum_equation.append(torch.ones((self._num_observations), device=self._device, dtype=torch.double))
            einsum_equation.append([1])

        # Each input used to calculate current output
        for input, index in inputs_with_indices:
            einsum_equation.append(input)
            einsum_equation.append([0, 1, index+2])

        # Cpts of the factor nodes
        einsum_equation.append(self._cpts)
        einsum_equation.append([0, *range(2, self._num_inputs+2)])

        # Desired output dimensions
        einsum_equation.append([0, 1, i_output+2])

        return einsum_equation
        
    def set_output_tensor(self, node: Node, output_node: Node, tensor: torch.Tensor):
        i_node = self.nodes.index(node)
        
        if output_node == node:
            i_output = -1
        else:
            i_output = self._parents[node].index(output_node)

        self._output_tensors[i_output][i_node] = tensor
        
    def get_input_tensor(self, node: Node, input_node: Node) -> torch.Tensor:
        i_node = self.nodes.index(node)
        
        if input_node == node:
            i_input = -1
        else:
            i_input = self._parents[node].index(input_node)

        return self._inputs[i_input][i_node]

    def get_node_inputs(self, node: Node) -> List[torch.Tensor]:
        node_index = self.nodes.index(node)

        return [
            input[node_index]
            for input
            in self._inputs
        ]

class FactorGraph:
    def __init__(self,
                 bayesian_network: BayesianNetwork,
                 num_observations: int,
                 observed_nodes: List[Node],
                 device: torch.device):
        self._device = device
        self._observed_nodes = observed_nodes
        self._local_likelihoods: torch.Tensor = torch.zeros((num_observations, len(bayesian_network.nodes)), dtype=torch.double, device=self._device)

        NodeGroupKey = namedtuple("NodeGroupKey", f'num_inputs num_states')
        
        # Instantiate variable node groups
        variable_node_groups_key_func = lambda node: NodeGroupKey(
            len(bayesian_network.children[node]) + 2 
                if node in self._observed_nodes 
                else len(bayesian_network.children[node]) + 1, 
            node.num_states)

        self.variable_node_groups = [
            VariableNodeGroup(
                nodes,
                [
                    self._local_likelihoods[:, bayesian_network.nodes.index(node)]
                    for node
                    in nodes
                ],
                bayesian_network.children,
                bayesian_network.parents,
                key.num_inputs,
                key.num_states,
                num_observations,
                self._device
            )
            for key, nodes
            in 
            [
                (key, list(nodes))
                for key, nodes
                in itertools.groupby(sorted(bayesian_network.nodes, key=variable_node_groups_key_func), key=variable_node_groups_key_func)
            ]
        ]

        # Instantiate factor node groups
        factor_node_groups_key_func = lambda node: node.cpt.shape
        self.factor_node_groups = [
            FactorNodeGroup(
                list(nodes),
                bayesian_network.children,
                bayesian_network.parents,
                len(key),
                list(key),
                num_observations,
                self._device
            )
            for key, nodes
            in itertools.groupby(sorted(bayesian_network.nodes, key=factor_node_groups_key_func), key=factor_node_groups_key_func)
        ]

        # Connect the node groups
        for node in bayesian_network.nodes:
            variable_node_group = self.get_variable_node_group(node)
            factor_node_group = self.get_factor_node_group(node)

            # ### Variable nodes
            # Corresponding factor node
            tensor = factor_node_group.get_input_tensor(node, node)
            variable_node_group.set_output_tensor(node, node, tensor)
            
            # Child factor nodes
            for child_node in bayesian_network.children[node]:
                child_factor_node_group = self.get_factor_node_group(child_node)
                tensor = child_factor_node_group.get_input_tensor(child_node, node)

                variable_node_group.set_output_tensor(node, child_node, tensor)

            # ### Factor nodes
            # Corresponding variable node
            tensor = variable_node_group.get_input_tensor(node, node)
            factor_node_group.set_output_tensor(node, node, tensor)

            # Parent variable nodes
            for parent_node in bayesian_network.parents[node]:
                parent_variable_node_group = self.get_variable_node_group(parent_node)
                tensor = parent_variable_node_group.get_input_tensor(parent_node, node)

                factor_node_group.set_output_tensor(node, parent_node, tensor)

    def get_variable_node_group(self, node: Node) -> VariableNodeGroup:
        [variable_node_group] = [variable_node_group for variable_node_group in self.variable_node_groups if node in variable_node_group.nodes]
        return variable_node_group
        
    def get_factor_node_group(self, node: Node) -> FactorNodeGroup:
        [factor_node_group] = [factor_node_group for factor_node_group in self.factor_node_groups if node in factor_node_group.nodes]
        return factor_node_group

    def iterate(self) -> None:
        for variable_node_group in self.variable_node_groups:
            variable_node_group.calculate_outputs()

        for factor_node_group in self.factor_node_groups:
            factor_node_group.calculate_outputs()

    def enter_evidence(self, evidence):
        for observed_node, observations in zip(self._observed_nodes, evidence):
            variable_node_group = self.get_variable_node_group(observed_node)
            variable_node_group.set_observations(observed_node, observations)
