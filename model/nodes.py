from abc import abstractmethod, ABC
from enum import Enum

import numpy as np


class NodeType(Enum):
    CPTNode = 1


class Node(ABC):
    @property
    @abstractmethod
    def node_type(self) -> NodeType:
        pass


class CPTNode(Node):
    node_type = NodeType.CPTNode

    def __init__(self, cpt: np.array):
        if not np.all(np.abs(cpt.sum(axis=-1) - 1) < 1e-15):
            raise Exception('The CPT should sum to 1 along the last dimension')

        self.numK = cpt.shape[-1]
        self.cpt = cpt

