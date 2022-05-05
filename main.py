import numpy as np
import torch

from common.utilities import Cfg
from inference_engines.factor_graph.factor_graph import FactorGraph
from inference_engines.torch_naive_inference_machine import TorchNaiveInferenceMachine
from inference_engines.torch_sum_product_algorithm_inference_machine import TorchSumProductAlgorithmInferenceMachine
from model.bayesian_network import BayesianNetwork
from model.nodes import CPTNode

torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### WARNING: networks with loops don't seem to work with SP yet. Seems a problem with normalization of messages
# and ever shrinking values after many iterations. Log-likelihood calculation is broken
# True network
Q1 = CPTNode(np.array([1 / 5, 4 / 5], dtype=np.float64), name='Q1')
Q2 = CPTNode(np.array([0.2, 0.5, 0.3], dtype=np.float64), name='Q2')
Q3 = CPTNode(np.array([[0.4, 0.6], [0.8, 0.2]], dtype=np.float64), name='Q3')
Q4 = CPTNode(np.array([[[1 / 6, 3 / 6, 2 / 6], [5 / 9, 3 / 9, 1 / 9], [7 / 11, 1 / 11, 3 / 11]],
                       [[7 / 13, 2 / 13, 4 / 13], [4 / 17, 5 / 17, 8 / 17], [12 / 19, 2 / 19, 5 / 19]]],
                      dtype=np.float64), name='Q4')
Y1 = CPTNode(np.array([[0.99, 0.01], [0.01, 0.99]], dtype=np.float64), name='Y1')
Y2 = CPTNode(np.array([[[0.99, 0.01/5, 0.01/5, 0.01/5, 0.01/5, 0.01/5], [0.01/5, 0.99, 0.01/5, 0.01/5, 0.01/5, 0.01/5], [0.01/5, 0.01/5, 0.99, 0.01/5, 0.01/5, 0.01/5]],
                       [[0.01/5, 0.01/5, 0.01/5, 0.99, 0.01/5, 0.01/5], [0.01/5, 0.01/5, 0.01/5, 0.01/5, 0.99, 0.01/5], [0.01/5, 0.01/5, 0.01/5, 0.01/5, 0.01/5, 0.99]]], dtype=np.float64), name='Y2')

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

evidence = torch.tensor([[0, 0]], device=torch_device, dtype=torch.int)

a = []
b = []
c = []
d = []
def callback(factor_graph: FactorGraph, iteration):
    a.append(factor_graph.factor_nodes[Q1].input_messages[0].get_value())
    b.append(factor_graph.factor_nodes[Q1].output_messages[0].get_value())
    c.append(factor_graph.factor_nodes[Q4].input_messages[2].get_value())
    d.append(factor_graph.factor_nodes[Q4].output_messages[2].get_value())

sp_inference_machine = TorchSumProductAlgorithmInferenceMachine(
    bayesian_network=network,
    observed_nodes=[Y1, Y2],
    device=torch_device,
    num_iterations=30,
    callback=callback,
    num_observations=len(evidence))
sp_inference_machine.enter_evidence(evidence)
ll0 = sp_inference_machine.log_likelihood()

cfg = Cfg({'device': 'cpu'})
naive_inference_machine = TorchNaiveInferenceMachine(
    cfg,
    bayesian_network=network,
    observed_nodes=[Y1, Y2])
naive_inference_machine.enter_evidence(evidence)
ll1 = naive_inference_machine.log_likelihood()

a = torch.squeeze(torch.stack(a))
b = torch.squeeze(torch.stack(b))
c = torch.squeeze(torch.stack(c))
d = torch.squeeze(torch.stack(d))

pass
