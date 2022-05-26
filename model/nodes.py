from abc import abstractmethod, ABC
from enum import Enum

import torch


class NodeType(Enum):
    CPTNode = 1


class Node(ABC):
    def __init__(self, name=None):
        self.name = name

    def __repr__(self):
        return super().__repr__() \
            if self.name is None \
            else f'{type(self).__name__} - {self.name}'

    @property
    @abstractmethod
    def node_type(self) -> NodeType:
        pass

class CPTNode(Node):
    node_type = NodeType.CPTNode

    def __init__(self, cpt: torch.tensor, name=None):
        super().__init__(name)

        if not torch.all(torch.abs(cpt.sum(axis=-1) - 1) < 1e-15):
            raise Exception('The CPT should sum to 1 along the last dimension')

        self.numK = cpt.shape[-1]
        self.cpt = cpt
        self.name = name

