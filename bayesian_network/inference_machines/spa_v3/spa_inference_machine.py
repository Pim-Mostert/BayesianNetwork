from dataclasses import dataclass
from typing import Callable, List, Optional

import networkx as nx
import torch

from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.inference_machines.abstractions import IInferenceMachine
from bayesian_network.inference_machines.common import (
    InferenceMachineSettings,
)
from bayesian_network.inference_machines.evidence import Evidence
from bayesian_network.inference_machines.spa_v3.factor_graph import FactorGraph


@dataclass
class SpaInferenceMachineSettings(InferenceMachineSettings):
    num_iterations: int | None = None
    allow_loops: bool = False
    callback: Optional[Callable[[int], None]] = None


class SpaInferenceMachine(IInferenceMachine):
    def __init__(
        self,
        settings: SpaInferenceMachineSettings,
        bayesian_network: BayesianNetwork,
        observed_nodes: List[Node],
        num_observations: int,
    ):
        self._settings = settings
        self.num_observations = num_observations
        self.observed_nodes = observed_nodes
        self.factor_graph = FactorGraph(
            bayesian_network=bayesian_network,
            observed_nodes=observed_nodes,
            torch_settings=self._settings.torch_settings,
            num_observations=self.num_observations,
        )

        self._num_iterations = self._settings.num_iterations
        if nx.is_tree(self.factor_graph.G):
            if not self._settings.num_iterations:
                self._num_iterations = nx.diameter(self.factor_graph.G)
        else:
            if not self._settings.allow_loops:
                raise ValueError(
                    "Network contains loops, but the specified settings disallow them."
                )

            if not self._settings.num_iterations:
                raise ValueError(
                    "Networks containers loops; number of iterations must be set explicitly."
                )

        self.must_iterate: bool = True

    @property
    def settings(self) -> InferenceMachineSettings:
        return self._settings

    def infer_single_nodes(self, nodes: List[Node]) -> List[torch.Tensor]:
        if self.must_iterate:
            self._iterate()

        return [self._infer_single_node(node) for node in nodes]

    def _infer_single_node(self, node: Node) -> torch.Tensor:
        variable_node_group = self.factor_graph.get_variable_node_group(node)
        factor_node_group = self.factor_graph.get_factor_node_group(node)

        variable_node_tensor = variable_node_group.get_input_tensor(node, node)
        factor_node_tensor = factor_node_group.get_input_tensor(node, node)

        p = variable_node_tensor * factor_node_tensor

        return p

    def infer_nodes_with_parents(self, child_nodes: List[Node]) -> List[torch.Tensor]:
        if self.must_iterate:
            self._iterate()

        return [self._infer_node_with_parents(node) for node in child_nodes]

    def _infer_node_with_parents(self, node: Node) -> torch.Tensor:
        factor_node_group = self.factor_graph.get_factor_node_group(node)

        einsum_equation = []
        for index, input in enumerate(factor_node_group.get_node_inputs(node)):
            einsum_equation.append(input)
            einsum_equation.append([0, index + 1])

        einsum_equation.append(factor_node_group.node_cpts[node])
        einsum_equation.append(range(1, factor_node_group._num_inputs + 1))

        einsum_equation.append(range(0, factor_node_group._num_inputs + 1))

        p = torch.einsum(*einsum_equation)

        return p

    def _iterate(self):
        for iteration in range(self._settings.num_iterations):
            self.factor_graph.iterate()

            if self._settings.callback:
                self._settings.callback(iteration)

        self.must_iterate = False

    def enter_evidence(self, evidence: Evidence):
        # evidence.shape: num_observed_nodes x [num_observations x num_states], one-hot encoded # noqa
        self.factor_graph.enter_evidence(evidence)

        self.must_iterate = True

    def log_likelihood(self) -> float:
        if not self.observed_nodes:
            raise Exception("Log likelihood can't be calculated with 0 observed nodes")

        if self.must_iterate:
            self._iterate()

        local_log_likelihoods = [
            variable_node_group.local_log_likelihoods
            for variable_node_group in self.factor_graph.variable_node_groups
        ]
        log_likelihoods = torch.cat(local_log_likelihoods)

        if self._settings.average_log_likelihood:
            return log_likelihoods.sum(dim=0).mean().item()
        else:
            return log_likelihoods.sum().item()
