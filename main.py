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



pass

x = torch.concat([a.flatten() for a in p_true])
y = torch.concat([a.flatten() for a in p])

import matplotlib.pyplot as plt
plt.scatter(x, y)

print(ll_true)
print(ll)

pass

#
#
# num_i = 100
#
# a1 = np.zeros((num_i, 2)); b1 = np.zeros((num_i, 2)); c1 = np.zeros((num_i))
# a2 = np.zeros((num_i, 2)); b2 = np.zeros((num_i, 2)); c2 = np.zeros((num_i))
# a3 = np.zeros((num_i, 2)); b3 = np.zeros((num_i, 2)); c3 = np.zeros((num_i))
# a4 = np.zeros((num_i, 2)); b4 = np.zeros((num_i, 2)); c4 = np.zeros((num_i))
# a5 = np.zeros((num_i, 2)); b5 = np.zeros((num_i, 2)); c5 = np.zeros((num_i))
# a7 = np.zeros((num_i, 2)); b7 = np.zeros((num_i, 2)); c7 = np.zeros((num_i))
# a8 = np.zeros((num_i, 2)); b8 = np.zeros((num_i, 2)); c8 = np.zeros((num_i))
# a9 = np.zeros((num_i, 2)); b9 = np.zeros((num_i, 2)); c9 = np.zeros((num_i))
# a10 = np.zeros((num_i, 2)); b10 = np.zeros((num_i, 2)); c10 = np.zeros((num_i))
# # a11 = np.zeros((num_i, 2)); b11 = np.zeros((num_i, 2)); c11 = np.zeros((num_i))
#
# a1[0] = np.array([0.5, 0.5]); b1[0] = np.array([0.5, 0.5]); c1[0] = 1
# a2[0] = np.array([0.5, 0.5]); b2[0] = np.array([0.5, 0.5]); c2[0] = 1
# a3[0] = np.array([0.5, 0.5]); b3[0] = np.array([0.5, 0.5]); c3[0] = 1
# a4[0] = np.array([0.5, 0.5]); b4[0] = np.array([0.5, 0.5]); c4[0] = 1
# a5[0] = np.array([0.5, 0.5]); b5[0] = np.array([0.5, 0.5]); c5[0] = 1
# a7[0] = np.array([0.5, 0.5]); b7[0] = np.array([0.5, 0.5]); c7[0] = 1
# a8[0] = np.array([0.5, 0.5]); b8[0] = np.array([0.5, 0.5]); c8[0] = 1
# a9[0] = np.array([0.5, 0.5]); b9[0] = np.array([0.5, 0.5]); c9[0] = 1
# a10[0] = np.array([0.5, 0.5]); b10[0] = np.array([0.5, 0.5]); c10[0] = 1
# # a11[0] = np.array([0.5, 0.5]); b11[0] = np.array([0.5, 0.5]); c11[0] = 1
#
# for i in range(1, num_i):
#     c1[i] = (b1[i-1] * b2[i-1] * b3[i-1]).sum()
#     a1[i] = (b2[i-1] * b3[i-1]) / c1[i]
#     b1[i] = Q1.cpt
#
#     c2[i] = (b1[i-1] * b3[i-1]).sum()
#     a2[i] = (b1[i-1] * b3[i-1]) / c2[i]
#     b2[i] = (Y2.cpt * a4[i-1][None, :]).sum(axis=1)
#
#     c3[i] = (b1[i-1] * b2[i-1]).sum()
#     a3[i] = (b1[i-1] * b2[i-1]) / c3[i]
#     b3[i] = (Q3.cpt * a5[i-1][None, :]).sum(axis=1)
#
#     c4[i] = b4[i-1].sum()
#     a4[i] = np.array([1, 0]) / c4[i]
#     b4[i] = (Y2.cpt * a2[i-1][:, None]).sum(axis=0)
#
#     c5[i] = (b5[i-1] * b7[i-1] * b8[i-1]).sum()
#     a5[i] = (b7[i-1] * b8[i-1]) / c5[i]
#     b5[i] = (Q3.cpt * a3[i-1][:, None]).sum(axis=0)
#
#     c7[i] = (b5[i-1] * b8[i-1]).sum()
#     a7[i] = (b5[i-1] * b8[i-1]) / c7[i]
#     b7[i] = (Y4.cpt * a9[i-1][None, :]).sum(axis=1)
#
#     c8[i] = (b5[i-1] * b7[i-1]).sum()
#     a8[i] = (b5[i-1] * b7[i-1]) / c8[i]
#     b8[i] = (Y5.cpt * a10[i-1][None, :]).sum(axis=1)
#
#     c9[i] = b9[i-1].sum()
#     a9[i] = np.array([0, 1]) / c9[i]
#     b9[i] = (Y4.cpt * a7[i-1][:, None]).sum(axis=1)
#
#     c10[i] = b10[i-1].sum()
#     a10[i] = np.array([0, 1]) / c10[i]
#     b10[i] = (Y5.cpt * a8[i-1][:, None]).sum(axis=0)
#
#     # c11[i] = (b1[i-1] * b2[i-1] * b3[i-1]).sum()
#     # a11[i] = (b1[i-1] * b2[i-1] * b3[i-1]) / c11[i]
#     # b11[i] = (Y4.cpt * a7[i-1][None, :, None] * a9[i-1][None, None, :]).sum(axis=(1, 2))
#
#     pass
#
# plt.figure()
# plt.subplot(3, 1, 1); plt.plot(range(num_i), np.concatenate([a1, a2, a3, a4, a5, a7, a8, a9, a10], axis=1))
# plt.subplot(3, 1, 2); plt.plot(range(num_i), np.concatenate([b1, b2, b3, b4, b5, b7, b8, b9, b10], axis=1))
# plt.subplot(3, 1, 3); plt.plot(range(num_i), np.stack([c1, c2, c3, c4, c5, c7, c8, c9, c10], axis=1))
#
# pass
