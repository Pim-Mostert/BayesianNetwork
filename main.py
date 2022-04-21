import numpy as np
import torch

from common.utilities import Cfg
from inference_engines.torch_sum_product_algorithm_inference_machine import TorchSumProductAlgorithmInferenceMachine
from model.bayesian_network import BayesianNetwork
from model.nodes import CPTNode

torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# True network
Q1 = CPTNode(np.array([1/5, 4/5], dtype=np.float64), name='Q1')
Q2 = CPTNode(np.array([0.2, 0.5, 0.3], dtype=np.float64), name='Q2')
Q3 = CPTNode(np.array([[0.4, 0.6], [0.5, 0.5]], dtype=np.float64), name='Q3')
Q4 = CPTNode(np.array([[1/3, 1/3, 1/3], [1/3, 1/3, 1/3], [1/3, 1/3, 1/3]], dtype=np.float64), name='Q4')
Y1 = CPTNode(np.array([[1, 0], [0, 1]], dtype=np.float64), name='Y1')
Y2 = CPTNode(np.array([[[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]], [[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]]], dtype=np.float64), name='Y2')

nodes = [Q1, Q2, Q3, Q4, Y1, Y2]
parents = {
    Q1: [],
    Q2: [],
    Q3: [Q1],
    Q4: [Q2],
    Y1: [Q3],
    Y2: [Q3, Q4]
}
network = BayesianNetwork(nodes, parents)

def callback(factor_graph):
    node = Q4

    variable_node = factor_graph.variable_nodes[node]
    factor_node = factor_graph.factor_nodes[node]

    [value_to_factor_node] = [
        message.get_value()
        for message
        in variable_node.output_messages
        if message.destination is factor_node
    ]
    [value_from_factor_node] = [
        message.get_value()
        for message
        in variable_node.input_messages
        if message.source is factor_node
    ]

    p = value_from_factor_node * value_to_factor_node

    p /= p.sum()
    pass

cfg = Cfg()
cfg.num_iterations = 10
cfg.device = torch_device
cfg.callback = callback
sp_inference_machine = TorchSumProductAlgorithmInferenceMachine(cfg, network, [Y1, Y2])

p = sp_inference_machine.infer([Q3])
pass