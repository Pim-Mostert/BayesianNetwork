import numpy as np
import torch
import matplotlib.pyplot as plt

from common.utilities import Cfg
from inference_engines.torch_naive_inference_machine import TorchNaiveInferenceMachine
from model.bayesian_network import BayesianNetwork
from model.nodes import CPTNode

torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### WARNING: networks with loops don't seem to work with SP yet. Seems a problem with normalization of messages
# and ever shrinking values after many iterations. Log-likelihood calculation is broken
# True network
Q1 = CPTNode(np.array([1 / 5, 4 / 5], dtype=np.float64), name='Q1')
Q3 = CPTNode(np.array([[0.25, 0.75], [0.15, 0.85]], dtype=np.float64), name='Q2')
Y2 = CPTNode(np.array([[0.7, 0.3], [0.1, 0.9]], dtype=np.float64), name='Y2')
Y4 = CPTNode(np.array([[2/11, 9/11], [5/8, 3/8]], dtype=np.float64), name='Y4')
Y5 = CPTNode(np.array([[5/13, 8/13], [4/7, 3/7]], dtype=np.float64), name='Y5')

nodes = [Q1, Q3, Y2, Y4, Y5]
parents = {
    Q1: [],
    Q3: [Q1],
    Y2: [Q1],
    Y4: [Q3],
    Y5: [Q3],
}
network = BayesianNetwork(nodes, parents)

evidence = torch.tensor([[0, 1, 1]], device=torch_device, dtype=torch.int)

cfg = Cfg({'device': 'cpu'})
naive_inference_machine = TorchNaiveInferenceMachine(
    cfg,
    bayesian_network=network,
    observed_nodes=[Y2, Y4, Y5])
naive_inference_machine.enter_evidence(evidence)
ll_true = naive_inference_machine.log_likelihood()
p_true = naive_inference_machine.infer_single_nodes([Q1, Q3])


num_i = 100

a1 = np.zeros((num_i, 2)); b1 = np.zeros((num_i, 2)); c1 = np.zeros((num_i))
a2 = np.zeros((num_i, 2)); b2 = np.zeros((num_i, 2)); c2 = np.zeros((num_i))
a3 = np.zeros((num_i, 2)); b3 = np.zeros((num_i, 2)); c3 = np.zeros((num_i))
a4 = np.zeros((num_i, 2)); b4 = np.zeros((num_i, 2)); c4 = np.zeros((num_i))
a5 = np.zeros((num_i, 2)); b5 = np.zeros((num_i, 2)); c5 = np.zeros((num_i))
a7 = np.zeros((num_i, 2)); b7 = np.zeros((num_i, 2)); c7 = np.zeros((num_i))
a8 = np.zeros((num_i, 2)); b8 = np.zeros((num_i, 2)); c8 = np.zeros((num_i))
a9 = np.zeros((num_i, 2)); b9 = np.zeros((num_i, 2)); c9 = np.zeros((num_i))
a10 = np.zeros((num_i, 2)); b10 = np.zeros((num_i, 2)); c10 = np.zeros((num_i))
# a11 = np.zeros((num_i, 2)); b11 = np.zeros((num_i, 2)); c11 = np.zeros((num_i))

a1[0] = np.array([0.5, 0.5]); b1[0] = np.array([0.5, 0.5]); c1[0] = 1
a2[0] = np.array([0.5, 0.5]); b2[0] = np.array([0.5, 0.5]); c2[0] = 1
a3[0] = np.array([0.5, 0.5]); b3[0] = np.array([0.5, 0.5]); c3[0] = 1
a4[0] = np.array([0.5, 0.5]); b4[0] = np.array([0.5, 0.5]); c4[0] = 1
a5[0] = np.array([0.5, 0.5]); b5[0] = np.array([0.5, 0.5]); c5[0] = 1
a7[0] = np.array([0.5, 0.5]); b7[0] = np.array([0.5, 0.5]); c7[0] = 1
a8[0] = np.array([0.5, 0.5]); b8[0] = np.array([0.5, 0.5]); c8[0] = 1
a9[0] = np.array([0.5, 0.5]); b9[0] = np.array([0.5, 0.5]); c9[0] = 1
a10[0] = np.array([0.5, 0.5]); b10[0] = np.array([0.5, 0.5]); c10[0] = 1
# a11[0] = np.array([0.5, 0.5]); b11[0] = np.array([0.5, 0.5]); c11[0] = 1

for i in range(1, num_i):
    c1[i] = (b1[i-1] * b2[i-1] * b3[i-1]).sum()
    a1[i] = (b2[i-1] * b3[i-1]) / c1[i]
    b1[i] = Q1.cpt

    c2[i] = (b1[i-1] * b3[i-1]).sum()
    a2[i] = (b1[i-1] * b3[i-1]) / c2[i]
    b2[i] = (Y2.cpt * a4[i-1][None, :]).sum(axis=1)

    c3[i] = (b1[i-1] * b2[i-1]).sum()
    a3[i] = (b1[i-1] * b2[i-1]) / c3[i]
    b3[i] = (Q3.cpt * a5[i-1][None, :]).sum(axis=1)

    c4[i] = b4[i-1].sum()
    a4[i] = np.array([1, 0]) / c4[i]
    b4[i] = (Y2.cpt * a2[i-1][:, None]).sum(axis=0)

    c5[i] = (b5[i-1] * b7[i-1] * b8[i-1]).sum()
    a5[i] = (b7[i-1] * b8[i-1]) / c5[i]
    b5[i] = (Q3.cpt * a3[i-1][:, None]).sum(axis=0)

    c7[i] = (b5[i-1] * b8[i-1]).sum()
    a7[i] = (b5[i-1] * b8[i-1]) / c7[i]
    b7[i] = (Y4.cpt * a9[i-1][None, :]).sum(axis=1)

    c8[i] = (b5[i-1] * b7[i-1]).sum()
    a8[i] = (b5[i-1] * b7[i-1]) / c8[i]
    b8[i] = (Y5.cpt * a10[i-1][None, :]).sum(axis=1)

    c9[i] = b9[i-1].sum()
    a9[i] = np.array([0, 1]) / c9[i]
    b9[i] = (Y4.cpt * a7[i-1][:, None]).sum(axis=1)

    c10[i] = b10[i-1].sum()
    a10[i] = np.array([0, 1]) / c10[i]
    b10[i] = (Y5.cpt * a8[i-1][:, None]).sum(axis=0)

    # c11[i] = (b1[i-1] * b2[i-1] * b3[i-1]).sum()
    # a11[i] = (b1[i-1] * b2[i-1] * b3[i-1]) / c11[i]
    # b11[i] = (Y4.cpt * a7[i-1][None, :, None] * a9[i-1][None, None, :]).sum(axis=(1, 2))

    pass

plt.figure()
plt.subplot(3, 1, 1); plt.plot(range(num_i), np.concatenate([a1, a2, a3, a4, a5, a7, a8, a9, a10], axis=1))
plt.subplot(3, 1, 2); plt.plot(range(num_i), np.concatenate([b1, b2, b3, b4, b5, b7, b8, b9, b10], axis=1))
plt.subplot(3, 1, 3); plt.plot(range(num_i), np.stack([c1, c2, c3, c4, c5, c7, c8, c9, c10], axis=1))

pass
