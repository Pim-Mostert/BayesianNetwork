import numpy as np
import torch
# import matplotlib.pyplot as plt

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

evidence = torch.tensor([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5]], device=torch_device, dtype=torch.int)

cfg = Cfg({'device': 'cpu'})
naive_inference_machine = TorchNaiveInferenceMachine(
    cfg,
    bayesian_network=network,
    observed_nodes=[Y1, Y2])
naive_inference_machine.enter_evidence(evidence)
ll_true = naive_inference_machine.log_likelihood()
p_true = naive_inference_machine.infer_single_nodes([Q1, Q2, Q3, Q4])
p_family_true = naive_inference_machine.infer_children_with_parents([Q3, Q4])

num_iterations=200
sp_inference_machine = TorchSumProductAlgorithmInferenceMachine(
    bayesian_network=network,
    observed_nodes=[Y1, Y2],
    device=torch.device('cpu'),
    num_iterations=num_iterations,
    num_observations=12,
    callback=lambda factor_graph, iteration: print(f'Finished iteration {iteration}/{num_iterations}'))

sp_inference_machine.enter_evidence(evidence)
ll = sp_inference_machine.log_likelihood()
p = sp_inference_machine.infer_single_nodes([Q1, Q2, Q3, Q4])
p_family = sp_inference_machine.infer_children_with_parents([Q3, Q4])



x = torch.concat([a.flatten() for a in p_family_true])
y = torch.concat([a.flatten() for a in p_family])

import matplotlib.pyplot as plt
plt.scatter(x, y)

pass
