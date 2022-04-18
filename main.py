import numpy as np
import torch

from common.utilities import Cfg
from inference_engines import torch_sum_product_algorithm_inference_machine
from inference_engines.torch_sum_product_algorithm_inference_machine import TorchSumProductAlgorithmInferenceMachine
from model.bayesian_network import BayesianNetwork
from model.nodes import CPTNode

torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# True network
Q1 = CPTNode(np.array([1/5, 4/5], dtype=np.float64), name='Q1')
Q2 = CPTNode(np.array([0.2, 0.5, 0.3], dtype=np.float64), name='Q2')
Q3 = CPTNode(np.array([[0.4, 0.6], [0.5, 0.5]], dtype=np.float64), name='Q3')
Q4 = CPTNode(np.array([[[1/3, 1/3, 1/3], [1/3, 1/3, 1/3]], [[1/3, 1/3, 1/3], [1/3, 1/3, 1/3]], [[1/3, 1/3, 1/3], [1/3, 1/3, 1/3]]], dtype=np.float64), name='Q4')
Y1 = CPTNode(np.array([[1, 0], [0, 1]], dtype=np.float64), name='Y1')
Y2 = CPTNode(np.array([[[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]], [[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]]], dtype=np.float64), name='Y2')

nodes = [Q1, Q2, Q3, Q4, Y1, Y2]
parents = {
    Q1: [],
    Q2: [],
    Q3: [Q1],
    Q4: [Q1, Q2],
    Y1: [Q3],
    Y2: [Q3, Q4]
}
network = BayesianNetwork(nodes, parents)

cfg = Cfg()
cfg.num_iterations = 1
sp_inference_machine = TorchSumProductAlgorithmInferenceMachine(cfg, network, [Y1, Y2])

sp_inference_machine.infer([])
pass