from typing import List

from common.utilities import Cfg
from model.bayesian_network import BayesianNetwork
from model.interfaces import IInferenceMachine
from model.nodes import Node


class TorchSumProductAlgorithmInferenceMachine(IInferenceMachine):
    def __init__(self, cfg: Cfg, bayesian_network: BayesianNetwork, observed_nodes: List[Node]):
        pass

    def infer(self, nodes: List[Node]):
        pass

    def enter_evidence(self, evidence):
        pass