import itertools
from collections import namedtuple
from typing import Dict, List

import torch
import torch.nn.functional as F

from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.common.torch_settings import TorchSettings
from bayesian_network.inference_machines.evidence import Evidence


class VariableNodeGroup:
    def __init__(
        self,
        nodes: List[Node],
        children: Dict[Node, List[Node]],
        parents: Dict[Node, List[Node]],
        num_inputs: int,
        num_states: int,
        num_observations: int,
        observed_nodes: List[Node],
        torch_settings: TorchSettings,
    ):
        self._torch_settings = torch_settings
        self.nodes = nodes
        self._children = children
        self._parents = parents
        self._num_nodes = len(self.nodes)
        self._num_observations = num_observations
        self._num_inputs = num_inputs
        self.num_states = num_states
        self._num_outputs = self._num_inputs

        if self._num_observations == 0:
            self._num_observations = 1

        # Placeholder
        self.local_log_likelihoods = torch.zeros(
            (self._num_nodes, self._num_observations),
            device=self._torch_settings.device,
            dtype=self._torch_settings.dtype,
        )

        self._i_observed_nodes = [
            self.nodes.index(observed_node) for observed_node in observed_nodes
        ]

        self._inputs = (
            torch.ones(
                (
                    self._num_inputs,
                    self._num_nodes,
                    self._num_observations,
                    self.num_states,
                ),
                device=self._torch_settings.device,
                dtype=self._torch_settings.dtype,
            )
            / num_states
        )

        self._output_tensors = [
            [
                # Placeholder
                torch.zeros(
                    (self._num_observations, self.num_states),
                    device=self._torch_settings.device,
                    dtype=self._torch_settings.dtype,
                )
                for _ in self.nodes
            ]
            for _ in range(self._num_outputs)
        ]

        self._i_output_to_local_factor_node = self._num_outputs - 1
        self._i_outputs_to_remote_factor_nodes = [
            i for i in range(self._num_outputs) if i != self._i_output_to_local_factor_node
        ]

        self._calculation_result = torch.empty(())  # Placeholder
        # output_tensor[0][0][:], output_tensor[0][1][:], ..., output_tensor[0][K],
        # output_tensor[1][0][:], output_tensor[1][1][:], ..., output_tensor[1][K],
        # ...
        # output_tensor[J][0][:], output_tensor[J][1][:], ..., output_tensor[J][K]
        #     = self._calculation_result.reshape[J * K, N, F]
        #
        #  J: Number of outputs
        #  K: Number of nodes
        #  N: Number of observations
        #  F: Number of states
        self._calculation_assignment_statement = (
            ", ".join(
                [
                    f"self._output_tensors[{i_output}][{i_node}][:]"
                    for i_output in range(self._num_outputs)
                    for i_node in range(self._num_nodes)
                ]
            )
            + " = self._calculation_result"
            + ".reshape(self._num_outputs*self._num_nodes, self._num_observations, self.num_states)"
        )

        self._calculation_indices_per_i_output = {
            i_output: [
                i_output_inner
                for i_output_inner in range(self._num_outputs)
                if i_output_inner != i_output
            ]
            for i_output in range(self._num_outputs)
        }

    def calculate_outputs(self):
        i_remote = self._i_outputs_to_remote_factor_nodes
        i_local = self._i_output_to_local_factor_node

        # [num_inputs, num_nodes, num_observations, num_states]
        x1 = self._inputs.log()

        # [1, num_nodes, num_observations, num_states]
        x2 = x1.sum(dim=0, keepdim=True)

        # [num_outputs, num_nodes, num_observations, num_states]
        x3 = x2 - x1

        # Normalization to remote factor nodes
        x3[i_remote] = F.softmax(x3[i_remote], dim=3)

        # [1, num_nodes, num_observations, 1]
        z = x2.max(dim=3, keepdim=True).values

        # c
        c = (x2 - z).exp().sum(dim=3, keepdim=True).log()

        x4 = x3[i_local][None, :, :, :] - z
        x4 = x4 - c
        x3[i_local] = x4.exp()

        self._calculation_result = x3

        # Assign calculation result to output vectors
        exec(self._calculation_assignment_statement)

        # Store local likelihoods
        self.local_log_likelihoods = (c + z).squeeze(dim=(0, 3))

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

    def set_observations(self, observations: torch.Tensor):
        # observations: torch.Tensor[num_nodes, num_observations, num_states]
        i_input = -2

        self._inputs[i_input, self._i_observed_nodes, :, :] = observations


class FactorNodeGroup:
    def __init__(
        self,
        nodes: List[Node],
        children: Dict[Node, List[Node]],
        parents: Dict[Node, List[Node]],
        num_inputs: int,
        inputs_num_states: List[int],
        num_observations: int,
        torch_settings: TorchSettings,
    ):
        self._torch_settings = torch_settings
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
            torch.ones(
                (
                    self._num_nodes,
                    self._num_observations,
                    self._inputs_num_states[i_input],
                ),
                device=self._torch_settings.device,
                dtype=self._torch_settings.dtype,
            )
            / torch.tensor(
                self._inputs_num_states[i_input],
                device=self._torch_settings.device,
                dtype=self._torch_settings.dtype,
            )
            for i_input in range(self._num_inputs)
        ]

        self.node_cpts = {node: self._cpts[i] for i, node in enumerate(self.nodes)}

        self._output_tensors = [
            [
                # Placeholder
                torch.zeros(
                    (
                        self._num_observations,
                        self._outputs_num_states[i_output],
                    ),
                    device=self._torch_settings.device,
                    dtype=self._torch_settings.dtype,
                )
                for _ in range(self._num_nodes)
            ]
            for i_output in range(self._num_outputs)
        ]

        # self._output_tensor[0][:], self._output_tensor[1][:], ..., self._output_tensor[self.num_nodes-1][:] # noqa
        self._calculation_output_tensor = torch.empty(())  # Placeholder
        self._calculation_result = torch.empty(())  # Placeholder
        self._calculation_assignment_statement = (
            ", ".join(
                [
                    f"self._calculation_output_tensor[{i_node}][:]"
                    for i_node in range(self._num_nodes)
                ]
            )
            + " = self._calculation_result"
        )

        self._calculation_einsum_equation_per_output = [
            self._construct_einsum_equation_for_output(i_output)
            for i_output in range(self._num_outputs)
        ]

    def calculate_outputs(self):
        for i_output in range(self._num_outputs):
            self._calculation_output_tensor = self._output_tensors[i_output]

            einsum_equation = self._calculation_einsum_equation_per_output[i_output]
            self._calculation_result = torch.einsum(*einsum_equation)

            exec(self._calculation_assignment_statement)

    def _construct_einsum_equation_for_output(self, i_output: int) -> List:
        # Get all inputs with indices, except for the input corresponding to current output # noqa
        inputs_with_indices = [
            (input, index)
            for input, index in zip(self._inputs, range(self._num_inputs))
            if index != i_output
        ]

        # Example einsum equation:
        #   'kna, knc, knd, kabcd->knb', input0, input2, input3, cpt
        #   [input0, [..., 0], input2, [..., 2], input3, [..., 3], cpt, [..., 0, 1, 2, 3], [..., 1]] # noqa
        #
        # k: num_nodes
        # n: num_observations
        einsum_equation = []

        if not inputs_with_indices:
            einsum_equation.append(
                torch.ones(
                    (self._num_observations),
                    device=self._torch_settings.device,
                    dtype=self._torch_settings.dtype,
                )
            )
            einsum_equation.append([1])

        # Each input used to calculate current output
        for input, index in inputs_with_indices:
            einsum_equation.append(input)
            einsum_equation.append([0, 1, index + 2])

        # Cpts of the factor nodes
        einsum_equation.append(self._cpts)
        einsum_equation.append([0, *range(2, self._num_inputs + 2)])

        # Desired output dimensions
        einsum_equation.append([0, 1, i_output + 2])

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

        return [input[node_index] for input in self._inputs]


class FactorGraph:
    def __init__(
        self,
        bayesian_network: BayesianNetwork,
        num_observations: int,
        observed_nodes: List[Node],
        torch_settings: TorchSettings,
    ):
        self._torch_settings = torch_settings
        self._observed_nodes = observed_nodes

        NodeGroupKey = namedtuple("NodeGroupKey", "num_inputs num_states")

        # Instantiate variable node groups
        def variable_node_groups_key_func(node):
            return NodeGroupKey(
                (
                    len(bayesian_network.children[node]) + 2
                    if node in self._observed_nodes
                    else len(bayesian_network.children[node]) + 1
                ),
                node.num_states,
            )

        self.variable_node_groups = [
            VariableNodeGroup(
                nodes,
                bayesian_network.children,
                bayesian_network.parents,
                key.num_inputs,
                key.num_states,
                num_observations,
                observed_nodes=[
                    observed_node
                    for observed_node in self._observed_nodes
                    if observed_node in set(nodes)
                ],
                torch_settings=self._torch_settings,
            )
            for key, nodes in [
                (key, list(nodes))
                for key, nodes in itertools.groupby(
                    sorted(
                        bayesian_network.nodes,
                        key=variable_node_groups_key_func,
                    ),
                    key=variable_node_groups_key_func,
                )
            ]
        ]

        # Instantiate factor node groups
        def factor_node_groups_key_func(node):
            return node.cpt.shape

        self.factor_node_groups = [
            FactorNodeGroup(
                list(nodes),
                bayesian_network.children,
                bayesian_network.parents,
                len(key),
                list(key),
                num_observations,
                torch_settings=self._torch_settings,
            )
            for key, nodes in itertools.groupby(
                sorted(bayesian_network.nodes, key=factor_node_groups_key_func),
                key=factor_node_groups_key_func,
            )
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
        [variable_node_group] = [
            variable_node_group
            for variable_node_group in self.variable_node_groups
            if node in variable_node_group.nodes
        ]
        return variable_node_group

    def get_factor_node_group(self, node: Node) -> FactorNodeGroup:
        [factor_node_group] = [
            factor_node_group
            for factor_node_group in self.factor_node_groups
            if node in factor_node_group.nodes
        ]
        return factor_node_group

    def iterate(self) -> None:
        for variable_node_group in self.variable_node_groups:
            variable_node_group.calculate_outputs()

        for factor_node_group in self.factor_node_groups:
            factor_node_group.calculate_outputs()

    def enter_evidence(self, all_evidence: Evidence):
        # evidence.data: List[(num_observed_nodes)], torch.Tensor: [num_observations, num_states] # noqa

        evidence_groups = (
            (variable_node_group, torch.stack(evidence))
            for variable_node_group, evidence in (
                (
                    variable_node_group,
                    [
                        evidence
                        for evidence, observed_node in zip(all_evidence.data, self._observed_nodes)
                        if observed_node in variable_node_group.nodes
                    ],
                )
                for variable_node_group in self.variable_node_groups
            )
            if evidence
        )

        for variable_node_group, evidence_group in evidence_groups:
            variable_node_group.set_observations(evidence_group)
