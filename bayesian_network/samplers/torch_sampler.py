from typing import Dict, List

import torch

from bayesian_network.bayesian_network import BayesianNetwork, Node
from bayesian_network.common.torch_settings import TorchSettings
from bayesian_network.interfaces import IBayesianNetworkSampler


class TorchBayesianNetworkSampler(IBayesianNetworkSampler):
    def __init__(self, bayesian_network: BayesianNetwork, torch_settings: TorchSettings):
        self.torch_settings = torch_settings
        self.bayesian_network = bayesian_network

        self.samplers: Dict[Node, NodeSampler] = {
            node: NodeSampler(node) for node in bayesian_network.nodes
        }

    def sample(self, num_samples: int, nodes: List[Node]) -> torch.Tensor:
        num_nodes = len(nodes)

        samples = torch.empty(
            (num_samples, num_nodes),
            device=self.torch_settings.device,
            dtype=torch.int32,
        )

        for i_sample in range(num_samples):
            samples[i_sample, :] = self._sample_single_trial(nodes)

        return samples

    def _sample_single_trial(self, nodes: List[Node]) -> torch.tensor:
        states = dict()

        for i, node in enumerate(nodes):
            states[node] = self._sample_single_node(node, states)

        return torch.tensor([states[node] for node in nodes], device=self.torch_settings.device)

    def _sample_single_node(self, node: Node, states: Dict[Node, torch.tensor]) -> torch.tensor:
        for parent in self.bayesian_network.parents[node]:
            if parent not in states:
                states[parent] = self._sample_single_node(parent, states)

        parent_states = torch.tensor(
            [states[parent] for parent in self.bayesian_network.parents[node]],
            device=self.torch_settings.device,
        )
        return self.samplers[node].sample(parent_states)


class NodeSampler:
    def __init__(self, node: Node):
        self.cpt = node.cpt

    def sample(self, parents_states: torch.tensor) -> torch.tensor:
        p = self.cpt[tuple(parents_states)]

        return torch.multinomial(p, 1, replacement=True).to(torch.int32)
