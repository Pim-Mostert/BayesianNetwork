from abc import abstractmethod, ABC
from typing import List, Dict, Type, Callable

import torch

import common as cmn
from common.interfaces import ISampler
from common.utilities import Cfg
from model.bayesian_network import BayesianNetwork
from model.nodes import Node, CPTNode


class TorchSampler(ISampler):
    def __init__(self, cfg0, bayesian_network: BayesianNetwork):
        self.device = cfg0.device

        cfg = Cfg()
        cfg.device = self.device
        self.samplers: Dict[Node, NodeSampler] = {node: NodeSamplers.create(cfg, node) for node in bayesian_network.nodes}
        self.parents = {node: bayesian_network.parents[node] for node in bayesian_network.nodes}

    def sample(self, num_samples: int, nodes: List[Node]) -> torch.Tensor:
        num_nodes = len(nodes)

        samples = torch.empty((num_samples, num_nodes), device=self.device, dtype=torch.int)

        for i_sample in range(num_samples):
            samples[i_sample, :] = self._sample_single_trial(nodes)

        return samples

    def _sample_single_trial(self, nodes: List[Node]) -> torch.tensor:
        states = dict()

        for (i, node) in enumerate(nodes):
            states[node] = self._sample_single_node(node, states)

        return torch.tensor([states[node] for node in nodes], device=self.device)

    def _sample_single_node(self, node: Node, states: Dict[Node, torch.tensor]) -> torch.tensor:
        for parent in self.parents[node]:
            if parent not in states:
                states[parent] = self._sample_single_node(parent, states)

        parent_states = torch.tensor([states[parent] for parent in self.parents[node]], device=self.device)
        return self.samplers[node].sample(parent_states)


class NodeSamplers:
    factories: Dict[Type, Callable] = {
        CPTNode: lambda cfg, node: CPTNodeSampler(cfg, node)
    }

    @staticmethod
    def create(cfg, node) -> 'NodeSampler':
        return NodeSamplers.factories[type(node)](cfg, node)


class NodeSampler(ABC):
    @abstractmethod
    def sample(self, parents_states: torch.tensor):
        pass


class CPTNodeSampler(NodeSampler):
    def __init__(self, cfg, cptnode: CPTNode):
        self.cpt = torch.tensor(cptnode.cpt, device=cfg.device)

    def sample(self, parents_states: torch.tensor) -> torch.tensor:
        p = self.cpt[tuple(parents_states)]

        return torch.multinomial(p, 1, replacement=True)
